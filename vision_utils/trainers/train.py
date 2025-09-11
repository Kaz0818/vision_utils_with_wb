from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from zoneinfo import ZoneInfo

# 注意：import 時の外部サービス初期化（wandb.login 等）は行わない。
# W&B の初期化は Notebook/スクリプト側で明示的に実施してください。
class Trainer:
    """
    モデルの訓練、評価、推論を管理するクラス。
    """

    def __init__(
    self,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    config: Dict,
    device: Optional[torch.device] = None,
    scheduler: Optional[_LRScheduler] = None,
    early_stopping_patience: Optional[int] = None,
    checkpoint_dir: str = "checkpoints",
    wandb_run: Optional["wandb.wandb_sdk.wandb_run.Run"] = None,
):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.config = config
        self.wandb_run = wandb_run

        self.timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime('%Y%m%d_%H%M%S')
        self.model_name = self.model.__class__.__name__

        # 追加: アーキ名を保存（configにarchが無ければクラス名を使う）
        self.arch = str(self.config.get("arch", self.model_name))

        # checkpoint_dirをRun×Archで分離（main側で既に分けているならそのまま同じ結果）
        base_ckpt_dir = Path(checkpoint_dir)
        if self.arch not in str(base_ckpt_dir):
            base_ckpt_dir = base_ckpt_dir / self.arch
        self.checkpoint_dir = base_ckpt_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        # 結果を格納するリスト
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

        # 最高のモデルのパス
        self.best_model_path: Optional[Path] = None

    def _train_one_epoch(self) -> Tuple[float, float]:
        """1エポック分の訓練を実行"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch+1}/{self.num_epochs}", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def _validate_one_epoch(self) -> Tuple[float, float]:
        """1エポック分の検証を実行"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def _infer_num_classes(self) -> Optional[int]:
        """モデルやconfigからnum_classesをできるだけ推定"""
        # 1) モデルにnum_classes属性がある場合
        if hasattr(self.model, "num_classes"):
            v = getattr(self.model, "num_classes")
            if isinstance(v, int) and v > 0:
                return v
        # 2) よくある最終層から推定
        def try_head(obj, path):
            cur = obj
            for p in path.split("."):
                if hasattr(cur, p):
                    cur = getattr(cur, p)
                else:
                    return None
            if hasattr(cur, "out_features"):
                return int(cur.out_features)
            return None
        for path in ["fc", "classifier", "head", "heads", "last_linear"]:
            n = try_head(self.model, path)
            if n is not None:
                return n
        # 3) configから
        v = self.config.get("num_classes")
        if isinstance(v, int) and v > 0:
            return v
        return None

    def _save_checkpoint(self, val_loss: float):
        """モデルのチェックポイントを保存"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            'model_state_dict': self.model.state_dict(),
            'arch': self.arch,                          # アーキを明示保存
            'num_classes': self._infer_num_classes(),   # 可能なら保存
            'epoch': self.current_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }

        best_filepath = self.checkpoint_dir / "best_model.pth"
        torch.save(state, best_filepath)
        self.best_model_path = best_filepath
        print(f"\n[INFO] Checkpoint saved: {best_filepath} (Val Loss: {val_loss:.4f})")

        # 任意: 人間に読みやすいファイル名も併存させたい場合は以下を有効化
        # pretty = self.checkpoint_dir / f"{self.arch}_best_valloss{val_loss:.4f}_ep{self.current_epoch+1:03d}.pth"
        # torch.save(state, pretty)

    def train(self) -> Dict[str, List[float]]:
        """訓練ループ全体を実行"""
        best_val_loss = float('inf')
        epochs_no_improve = 0

        if not self.wandb_run:
            print("[WARNING] W&B run not provided. Logging will be disabled.")

        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch

                train_loss, train_acc = self._train_one_epoch()
                val_loss, val_acc = self._validate_one_epoch()

                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                if self.wandb_run:
                    self.wandb_run.log({
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "train/acc": train_acc,
                        "val/acc": val_acc
                    })

                print(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

                if self.scheduler:
                    self.scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    self._save_checkpoint(val_loss)
                else:
                    epochs_no_improve += 1

                if self.early_stopping_patience and epochs_no_improve >= self.early_stopping_patience:
                    print(f"\n[INFO] Early stopping triggered after {self.early_stopping_patience} epochs with no improvement.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user.")
        finally:
            print("\n[INFO] Training finished.")

            # 学習完了時に最高のモデルをW&Bにアップロード
            if self.wandb_run and self.best_model_path and self.best_model_path.exists():
                # base_pathは親ディレクトリを指定（W&B上の相対構造を綺麗にするため）
                self.wandb_run.save(str(self.best_model_path), base_path=str(self.checkpoint_dir.parent))
                print(f"[INFO] Uploaded best model to W&B: {self.best_model_path.name}")

            # self.writer を使っていないためクローズのみログ
            print("[INFO] TensorBoard writer closed.")

        return self.history

    def get_best_model_path(self) -> Optional[Path]:
        return self.best_model_path

    def predict(
        self,
        dataloader: DataLoader,
        return_probs: bool = False
    ) -> Union[List[int], Tuple[List[int], List[List[float]]]]:
        """データローダーからのデータに対して推論を実行"""
        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc="Predicting", leave=False):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().tolist())

                if return_probs:
                    probs = torch.softmax(outputs, dim=1)
                    probabilities.extend(probs.cpu().tolist())

        if return_probs:
            return predictions, probabilities
        return predictions
