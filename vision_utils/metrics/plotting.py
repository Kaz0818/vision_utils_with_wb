from __future__ import annotations
from pathlib import Path
from typing import List, TYPE_CHECKING, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np  # ★ 追加：型ゆらぎ対策で使用

# wandb はオプショナル依存にする
try:
    import wandb as _wandb  # 実行時に使用
    WANDB_AVAILABLE = True
except Exception:
    _wandb = None  # type: ignore
    WANDB_AVAILABLE = False

if TYPE_CHECKING:  # 型チェック時のみ正確な型を使う
    import wandb

class Visualizer:
    def __init__(self, wandb_run: "wandb.wandb_sdk.wandb_run.Run", writer: Any | None = None):
        self.writer = writer
        
        if not wandb_run:
            raise ValueError("wandb.Run object must be provided.")
        if not WANDB_AVAILABLE:
            raise ImportError("wandb がインストールされていないため Visualizer は使用できません。'pip install wandb' または環境変数 WANDB_DISABLED=true をご検討ください。")
        self.wandb_run = wandb_run
        
        # wandbの実行ディレクトリ配下に成果物用のディレクトリを作成
        self.results_dir = Path(self.wandb_run.dir) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Visualizer] Artifacts will be saved to: {self.results_dir}")
    
    def run_evaluation(
        self,
        model: nn.Module,
        model_path: Path,
        dataloader: DataLoader,
        class_names: List[str],
        device: torch.device,
    ) -> None:
        """ 
        モデルの評価、プロット、W&Bへのログ記録をすべて実行する
        """
        print("\n--- [Visualizer] Starting Final Evaluation ---")
        
        # 1. モデルの重みを読み込む
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            print(f"[Visualizer] Loaded model checkpoint from: {model_path}")
        elif model_path is None:
            model.to(device)
            model.eval()
        else:
            print(f"[Visualizer] ERROR: Model path not found: {model_path}")
            return
        
        # 2. 予測と正解ラベルを収集
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="[Visualizer] Predicting"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # ★ numpy配列をリスト化して型のズレを最小化
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        
        # 3. Confusion MatrixとClassification Reportを生成・保存・アップロード
        self.log_confusion_matrix(all_labels, all_preds, class_names)
        self.log_classification_report(all_labels, all_preds, class_names)
        
        print("🚀 [Visualizer] Evaluation and plotting complete.")
        
    def log_confusion_matrix(self, y_true: List[int], y_pred: List[int], class_names: List[str]):
        """Confusion Matrixを作成し、W&Bにログとして記録する"""
        K = len(class_names)
        labels_idx = list(range(K))
        # ★ 重要：全クラス固定で行列サイズを K×K にする
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
        cm_path = self.results_dir / "confusion_matrix.png"
        
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        plt.figure(figsize=(12, 9))
        sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        
        # W&Bに画像をログとして記録
        if WANDB_AVAILABLE and _wandb is not None:
            self.wandb_run.log({"evaluation/confusion_matrix": _wandb.Image(str(cm_path))})
        # ファイル自体も保存
        self.wandb_run.save(str(cm_path), base_path=self.wandb_run.dir)
        print("[Visualizer] Logged confusion matrix to W&B.")
        
        
    def log_classification_report(self, y_true: List[int], y_pred: List[int], class_names: List[str]):
        """Classification Reportを作成し、W&Bにログとして記録する"""
        K = len(class_names)
        labels_idx = list(range(K))
        # ★ 重要：CMと同じ labels を指定＋未出現クラスは zero_division=0 で安全に
        report_str = classification_report(
            y_true, y_pred,
            labels=labels_idx,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        report_path = self.results_dir / "classification_report.txt"
        report_path.write_text(report_str)
        
        print("\nClassification Report:")
        print(report_str)

        # ファイルをW&Bに保存
        self.wandb_run.save(str(report_path), base_path=self.wandb_run.dir)
        print("[Visualizer] Logged classification report to W&B.")
