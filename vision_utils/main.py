import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import omegaconf as oc

from data.datamodule import make_transform
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

    
    dataset = datasets.MNIST(root=cfg.data.data_dir, train=True, download=True, transform=None)
    
    # 全インデックスとラベルを取得
    indices = list(range(len(dataset)))
    targets = dataset.targets.numpy() # MNISTは.targetsにラベルが入っている
    

    # stratifyにラベルを渡すことでクラス比率を維持した分割ができる
    train_idx, val_idx = train_test_split(
        indices,
        test_size=1 - cfg.data.split_ratio,
        stratify=targets,
        random_state=cfg.seed,
    )
    
    # transformを分けたdatasetを２つ用意
    train_dataset_full = datasets.MNIST(root=cfg.data.data_dir, train=True, download=True, transform=train_transform)
    val_dataset_full = datasets.MNIST(root=cfg.data.data_dir, train=True, download=True, transform=val_transform)
    
    # インデックスで絞り込み
    train_dataset = Subset(train_dataset_full, train_idx)
    val_dataset = Subset(val_dataset_full, val_idx)

    # DataLoader 作成
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=True)


    # testデータを読み込む
    test_dataset = datasets.FashionMNIST(root=cfg.data.data_dir, train=False, download=True, transform=val_transform)
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
    
if __name__ == "__main__":
    main()
    

#     # ===============推論 & 評価(Test Accuracy)===========================
#     # 推論（ラベル予測）
#     preds = trainer.predict(test_loader)

#     # 正解ラベル取得
#     true_labels = [label for _, label in test_dataset]

#     # 正解率の計算
#     correct = sum(p == t for p, t in zip(preds, true_labels))
#     acc = correct / len(true_labels)
#     print(f"[Test Accuracy] {acc * 100:.2f}%")


#     # ==========学習済みのモデルをロードして学習========================================
#     # モデルを再構築（構造は事前に一致させること！）
#     # model = SimpleCNN(num_classes=cfg["models"]["num_classes"])
#     model = get_model(cfg['models']['name'], num_classes=cfg['models']['num_classes'])
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])

#     # 保存済みモデルの読み込み
#     model, optimizer, loaded_epoch, val_loss = load_model(model, optimizer, path="checkpoints/best_model.pth")

#     # 推論テスト
#     model.eval()
#     model.to("cpu")
#     test_preds = trainer.predict(test_loader)  # trainerを再利用




#     cm_path = f"results/cm_epoch{cfg['train']['epochs']}.png"
#     trainer.plot_confusion_matrix_display(
#         model=trainer.model,
#         dataloader=val_loader,
#         class_names=val_loader.dataset.dataset.classes,
#         cm_save_path=cm_path,
#         epoch=1
#     )

# main()