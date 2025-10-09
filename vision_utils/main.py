import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import omegaconf as oc

from metrics.plotting import Visualizer
from data.datamodule import make_transform, build_dataset, get_targets_array
from trainers.train import Trainer
from grad_cam.grad_cam import GradCAMVisualizer
# --- 追加: ロギング初期化関数 ---

def setup_wandb(cfg: DictConfig):
    if not OmegaConf.select(cfg, "logging.wandb.enabled", default=False):
        return None
    import wandb, os

    wandb_dir = OmegaConf.select(cfg, "logging.wandb.dir", default="./")
    os.makedirs(wandb_dir, exist_ok=True)

    # 既存Runが生きていたら先に閉じる（reinitを使わない）
    if getattr(wandb, "run", None) is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    run_name = OmegaConf.select(cfg, "logging.wandb.name", default=None) \
               or datetime.now().strftime("%Y%m%d-%H%M%S")

    return wandb.init(
        project=OmegaConf.select(cfg, "logging.wandb.project", default="portfolio"),
        entity=OmegaConf.select(cfg, "logging.wandb.entity",  default=None),
        name=run_name,
        mode=OmegaConf.select(cfg, "logging.wandb.mode",     default="online"),
        dir=wandb_dir,  # 末尾に /wandb は付けない（W&Bが自動で作る）
        config=OmegaConf.to_container(cfg, resolve=True),
    )

def setup_mlflow(cfg: DictConfig):
    """logging.mlflow.* を参照。HydraConfig に依存しない安全版。"""
    if not OmegaConf.select(cfg, "logging.mlflow.enabled", default=False):
        return None

    import mlflow

    uri = OmegaConf.select(cfg, "logging.mlflow.tracking_uri", default="file:./mlruns")
    # file:// のときはディレクトリを確実に作る
    if uri.startswith("file://"):
        Path(uri.replace("file://", "")).mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(OmegaConf.select(cfg, "logging.mlflow.experiment", default="portfolio"))

    run_name = OmegaConf.select(cfg, "logging.mlflow.run_name", default=None) \
               or datetime.now().strftime("%Y%m%d-%H%M%S")

    mlflow.start_run(run_name=run_name)

    # 任意：主要パラメータを記録（.get ではなく select で安全に）
    mlflow.log_params({
        "epochs":        OmegaConf.select(cfg, "train.epochs", default=None),
        "batch_size":    OmegaConf.select(cfg, "data.batch_size", default=None),
        "optimizer":     OmegaConf.select(cfg, "optimizer._target_", default=str(OmegaConf.select(cfg, "optimizer"))),
        "model":         OmegaConf.select(cfg, "model._target_", default=str(OmegaConf.select(cfg, "model"))),
    })
    return mlflow



def resolve_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ============hydra設定===========================
@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print("Config:\n", oc.OmegaConf.to_yaml(cfg))
    
    print("=== keys at root ===", list(cfg.keys()))
    print("=== has wandb? ===", "wandb" in cfg)
    
    print("=== full cfg ===\n", oc.OmegaConf.to_yaml(cfg))

    
    # ============ Loggingを初期化 ============
    
    wandb_run = setup_wandb(cfg)
    mlflow_mod = setup_mlflow(cfg)      # 使うときは mlflow_mod.log_metric(...) で
    
    # ============データ準備 ==========================
    
    train_transform, val_transform = make_transform(0.5, 0.5)

    dataset = build_dataset(cfg.data.name, cfg.data.root, cfg.data.train)
    print("読み込み完了")
    # 全インデックスとラベルを取得
    indices = list(range(len(dataset)))
    targets = get_targets_array(dataset)
        
    # stratifyにラベルを渡すことでクラス比率を維持した分割ができる
    train_idx, val_idx = train_test_split(
        indices,
        test_size=1 - cfg.data.split_ratio,
        stratify=targets,
        random_state=cfg.seed,
    )
    
    # transformを分けたdatasetを２つ用意
    train_dataset_full = build_dataset(cfg.data.name, cfg.data.root, cfg.data.train, transform=train_transform)
    val_dataset_full = build_dataset(cfg.data.name, cfg.data.root, cfg.data.train, transform=val_transform)
    
    # インデックスで絞り込み
    train_dataset = Subset(train_dataset_full, train_idx)
    val_dataset = Subset(val_dataset_full, val_idx)

    # DataLoader 作成
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=True)


    # testデータを読み込む
    test_dataset = build_dataset(cfg.data.name, cfg.data.root, False, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    # ===============コンポーネント生成(_target_ -> instantiate)=================================
    model = instantiate(cfg.model)
    criterion = instantiate(cfg.criterion)
    
    # Optimizerはpartial_: trueなのでここでparamsを注入
    # optimizer = instantiate(cfg.optimizer, params=model.parameters())
    opt_ctor = instantiate(cfg.optimizer)
    optimizer = opt_ctor(model.parameters())

    # Trainerもinstantiate, DataLoaderやmodel等は引数で渡す
    device = resolve_device(name=cfg.train.device)
    print("=== cfg.trainer ===\n", oc.OmegaConf.to_yaml(cfg.trainer))

    trainer: Trainer = instantiate(
        cfg.trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        wandb_run=wandb_run,
    )
    # Training
    history = trainer.train()
    
    # === 例: MLflowに最終metricsを記録(任意)===
    if mlflow_mod:
        # 代表値だけでOK。履歴があるなら最後をログ
        mlflow_mod.log_metrics({
            "final/train_loss": history["train_loss"][-1],
            "final/val_loss": history["val_loss"][-1],
            "final/train_acc": history["train_acc"][-1],
            "final/val_acc": history["val_acc"][-1],
        })

    # ============Metrics=============
    print("start metrics")
    class_names = train_dataset_full.classes
    vis = Visualizer(wandb_run=wandb_run)
    vis.run_evaluation(model, model_path=None, dataloader=test_loader, class_names=class_names, device=device)
    print("Done metrics")
    
    # =======GradCAM==========
    print("start gradcam")
    grad_cam = GradCAMVisualizer(wandb_run=wandb_run, model=model, model_path=None, class_names=class_names)
    grad_cam.plot_random_samples(test_loader)
    print("done gradcam")
    
    # 片付け
    if mlflow_mod:
        import mlflow
        mlflow.end_run()
    if wandb_run:
        wandb_run.finish()
if __name__ == "__main__":
    main()
    