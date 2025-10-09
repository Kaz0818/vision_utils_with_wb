import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import omegaconf as oc

from metrics.plotting import Visualizer
from data.datamodule import make_transform, build_dataset, get_targets_array
from trainers.train import Trainer

def resolve_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ============hydra設定===========================
@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print("Config:\n", oc.OmegaConf.to_yaml(cfg))
    
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
        device=device
    )
    trainer.train()
    predicted = trainer.predict(test_loader)
    
    # ============Metrics=============
    vis = Visualizer()
    vis.run_evaluation(model, dataloader=test_loader, device=device)
    
if __name__ == "__main__":
    main()
    