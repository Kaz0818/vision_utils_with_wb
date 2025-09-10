import yaml
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from utils.data_utils import split_dataset
from models.model_cnn import SimpleCNN, SimpleCNNWithKaiming
from trainers.train import Trainer
from utils.save_load import save_model, load_model
from utils.gradcam_utils import apply_gradcam
from models.model_selector import get_model

# ========= Config読み取り・シード設定　=================
# config 読み込み
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 乱数シード固定（再現性）
torch.manual_seed(cfg["train"]["seed"])

# ============データ準備 ==========================
# データ変換
transform = transforms.ToTensor()

# データセット読み込み（train=True のみ）
dataset = datasets.FashionMNIST(root=cfg["data"]["data_dir"], train=True, download=True, transform=transform)

# データ分割
train_dataset, val_dataset = split_dataset(dataset,
                                           split_ratio=cfg["data"]["split_ratio"],
                                           seed=cfg["train"]["seed"])

# DataLoader 作成
train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False)


# testデータを読み込む
test_dataset = datasets.FashionMNIST(root=cfg["data"]["data_dir"], train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=cfg['train']['batch_size'], shuffle=False)

# ===========モデル構築(SimpleCNN or ResNet) ====================================

# model = SimpleCNN(num_classes=cfg["models"]["num_classes"])
# model = SimpleCNN(num_classes=cfg["models"]["num_classes"])
# model = SimpleCNNWithKaiming(num_classes=cfg['models']['num_classes'])

# モデル構築(転移学習)
model = get_model(cfg['models']['name'], num_classes=cfg['models']['num_classes'])


# ===============損失関数と最適化とTrainerの初期化=================================
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])

trainer = Trainer(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=cfg["train"]["epochs"],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path="checkpoints/",
    writer=None,
    early_stopping=cfg["train"].get("early_stopping", None)
)
trainer.train()

# ===============推論 & 評価(Test Accuracy)===========================
# 推論（ラベル予測）
preds = trainer.predict(test_loader)

# 正解ラベル取得
true_labels = [label for _, label in test_dataset]

# 正解率の計算
correct = sum(p == t for p, t in zip(preds, true_labels))
acc = correct / len(true_labels)
print(f"[Test Accuracy] {acc * 100:.2f}%")


# ==========学習済みのモデルをロードして学習========================================
# モデルを再構築（構造は事前に一致させること！）
# model = SimpleCNN(num_classes=cfg["models"]["num_classes"])
model = get_model(cfg['models']['name'], num_classes=cfg['models']['num_classes'])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])

# 保存済みモデルの読み込み
model, optimizer, loaded_epoch, val_loss = load_model(model, optimizer, path="checkpoints/best_model.pth")

# 推論テスト
model.eval()
model.to("cpu")
test_preds = trainer.predict(test_loader)  # trainerを再利用



# ================= Grad-CAMで1枚の画像を表示========================

# # Grad-CAM 対象の画像を1枚取得
# img, label = test_dataset[0]  # PIL画像 -> tensor（[1, 28, 28]）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 最後の conv 層を取得（SimpleCNN の conv2）
# target_layer = model.net[3]

# # Grad-CAM 実行
# timestamp_grad_cam = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# model.to(device)
# cam, class_idx = apply_gradcam(model, img, target_layer, device=device)

# # 可視化と保存
# img_np = img.squeeze().numpy()
# plt.figure(figsize=(5,5))
# plt.imshow(img_np, cmap='gray')
# plt.imshow(cam, cmap='jet', alpha=0.5)
# plt.title(f"Label: {label}, Pred: {class_idx}")
# plt.axis('off')
# os.makedirs('outputs', exist_ok=True)
# plt.savefig(f"outputs/{timestamp_grad_cam}_gradcam_result.png", bbox_inches='tight')
# plt.show()
# plt.close()



# =================Grad-CAMで複数画像を表示　=========================================
# 出力ディレクトリ作成
# os.makedirs("outputs", exist_ok=True)

# # test画像から16枚だけ取得
# sample_imgs, sample_labels = next(iter(test_loader))
# sample_imgs = sample_imgs[:16]
# sample_labels = sample_labels[:16]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()
# timestamp_grad_cam = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# # Grad-CAM適用して画像リストを生成
# fig, axes = plt.subplots(4, 4, figsize=(12, 12))

# for idx, ax in enumerate(axes.flat):
#     img = sample_imgs[idx]
#     label = sample_labels[idx].item()

#     # Grad-CAM 実行
#     cam, pred = apply_gradcam(model, img, target_layer=model.layer4[1].conv2, device=device)
#     # cam, pred = apply_gradcam(model, img, target_layer=model.net[3], device=device)

#     # 画像変換（白黒＋ヒートマップ）
#     img_np = img.squeeze().cpu().numpy()
#     ax.imshow(img_np, cmap='gray')
#     ax.imshow(cam, cmap='jet', alpha=0.5)
#     ax.set_title(f"GT: {label} / Pred: {pred}", fontsize=9)
#     ax.axis('off')

# plt.tight_layout()

# plt.savefig(f"outputs/{timestamp_grad_cam}_gradcam_batch.png")
# plt.show()
# plt.close()


# # gridを作成（normalizeで見やすく）
# grid = make_grid(sample_imgs, nrow=4, normalize=True)

# writer = SummaryWriter(log_dir="logs/tensorboard_logs")
# writer.add_image("Test Images", grid)
# writer.close()


# # ====================== GradCAMで間違えた画像だけ表示する ==========================

# # 保存ディレクトリ作成
# os.makedirs("outputs", exist_ok=True)

# # 1. 予測
# preds = trainer.predict(test_loader)
# true_labels = [label for _, label in test_dataset]

# # 2. 誤分類だけ抽出
# misclassified = [(img, pred, label)
#                  for (img, label), pred in zip(test_dataset, preds)
#                  if pred != label]

# # 3. 上位 N 件だけ表示
# N = 16
# misclassified = misclassified[:N]

# # 4. 描画
# fig, axes = plt.subplots(4, 4, figsize=(12, 12))

# for i, (img, pred, label) in enumerate(misclassified):
#     ax = axes[i // 4][i % 4]
#     ax.imshow(img.squeeze(), cmap="gray")
#     ax.set_title(f"GT: {label} / Pred: {pred}", fontsize=9)
#     ax.axis("off")

# plt.tight_layout()
# plt.savefig(f"outputs/{timestamp_grad_cam}_misclassified.png")
# plt.show()



# # ===========GradCAMで誤分類を表示================
# # 出力フォルダ作成
# os.makedirs("outputs", exist_ok=True)

# # 最大表示数
# N = 16
# misclassified = misclassified[:N]

# # Grad-CAM付きで描画
# fig, axes = plt.subplots(4, 4, figsize=(12, 12))

# for i, (img, pred, label) in enumerate(misclassified):
#     ax = axes[i // 4][i % 4]

#     # Grad-CAM 実行（img: Tensor [1,28,28]）
#     cam, _ = apply_gradcam(model, img, target_layer=model.layer4[1].conv2, class_idx=pred, device=device)

#     # 表示用画像
#     img_np = img.squeeze().cpu().numpy()
#     ax.imshow(img_np, cmap='gray')
#     ax.imshow(cam, cmap='jet', alpha=0.5)
#     ax.set_title(f"GT: {label} / Pred: {pred}", fontsize=9)
#     ax.axis("off")

# plt.tight_layout()
# plt.savefig(f"outputs/{timestamp_grad_cam}_misclassified_gradcam.png")
# plt.show()





# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def plot_confusion_matrix_display(model, dataloader, class_names, device=device, normalize=True, save_path=None):
    
#     model.eval()
#     y_true = []
#     y_pred = []
    

#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             outputs = model(x)
#             preds = torch.argmax(outputs, dim=1)
#             y_true.extend(y.cpu().tolist())
#             y_pred.extend(preds.cpu().tolist())

#     # 混同行列を生成
#     cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

#     # 混同行列を表示
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
#     plt.title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path)
#         print(f"[INFO] Saved confusion matrix to {save_path}")
#     else:
#         plt.show()

#     plt.close()



cm_path = f"results/cm_epoch{cfg['train']['epochs']}.png"
trainer.plot_confusion_matrix_display(
    model=trainer.model,
    dataloader=val_loader,
    class_names=val_loader.dataset.dataset.classes,
    cm_save_path=cm_path,
    epoch=1
)