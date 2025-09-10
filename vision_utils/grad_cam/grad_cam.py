import os
from typing import List, Optional, Tuple
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------------
# ユーティリティ
# ---------------------------
def is_vit_model(model: nn.Module) -> bool:
    m = unwrap_model(model)
    return hasattr(m, "patch_embed") and hasattr(m, "blocks")

def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)

def vit_reshape_transform(tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    ViT用: トークン列 [B, N, C] -> 2次元特徴 [B, C, H, W] へ。
    CLSトークンを除外し、grid_size が無い/合わない時も破綻しないよう保険。
    """
    if tensor.dim() != 3:
        return tensor
    B, N, C = tensor.shape
    m = unwrap_model(model)
    try:
        gh, gw = m.patch_embed.grid_size
    except Exception:
        side = int(round(math.sqrt(max(N - 1, 1))))
        gh, gw = side, side
    if gh * gw != max(N - 1, 1):
        side = int(round(math.sqrt(max(N - 1, 1))))
        gh, gw = side, side
    x = tensor[:, 1:, :]
    x = x.reshape(B, gh, gw, C).permute(0, 3, 1, 2).contiguous()
    return x

def denormalize_image(tensor: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    """
    正規化済み [C,H,W] を [H,W,3] の 0-1 float32 へ。1ch は 3ch に複製。
    """
    t = tensor.detach().cpu().clone().float()
    if t.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape: {tuple(t.shape)}")
    if t.size(0) == 1:  # グレースケール保険
        t = t.repeat(3, 1, 1)
    for ch, m, s in zip(t, mean, std):
        ch.mul_(s).add_(m)
    image_np = t.permute(1, 2, 0).numpy()
    return np.clip(image_np, 0, 1).astype(np.float32)

def find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    m = unwrap_model(model)
    last_conv = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            last_conv = mod
    return last_conv

# ---------------------------
# メインクラス
# ---------------------------
class GradCAMVisualizer:
    """
    Grad-CAM の計算と可視化（ViT/CNN対応、読みやすさ優先）
    """

    def __init__(
        self,
        wandb_run: wandb.wandb_sdk.wandb_run.Run, 
        model: nn.Module,
        model_path: Path,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        target_layer: Optional[nn.Module] = None,
    ):
        if not wandb_run:
            raise ValueError("wandb.Run object must be provided.")
        self.wand_run = wandb_run

        # device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 結果ディレクトリ
        self.results_dir = Path(self.wand_run.dir) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[GradCAMVisualizer] Artifacts will be saved to: {self.results_dir}")

        # モデルロード（厳密すぎて落ちないよう strict=False）
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(f"[GradCAMVisualizer] Model path not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[GradCAMVisualizer] load_state_dict: missing={missing}, unexpected={unexpected}")
        print(f"[GradCAMVisualizer] Loaded model checkpoint from: {model_path}")

        self.model = model.eval().to(self.device)

        self.class_names = class_names
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.is_vit = is_vit_model(self.model)
        self.target_layer = target_layer or self._select_default_target_layer()

        # ViT のときに使う reshape_transform を保持
        self.reshape_transform = (lambda t: vit_reshape_transform(t, self.model)) if self.is_vit else None

    def _select_default_target_layer(self) -> nn.Module:
        m = unwrap_model(self.model)
        if self.is_vit:
            for getter in (
                lambda mm: mm.blocks[-1].norm1,
                lambda mm: mm.blocks[-1].mlp.fc2,
                lambda mm: mm.blocks[-1],
            ):
                try:
                    layer = getter(m)
                    if layer is not None:
                        return layer
                except Exception:
                    continue
            raise ValueError("ViTのターゲット層が特定できませんでした。target_layer を指定してください。")
        else:
            last_conv = find_last_conv_layer(m)
            if last_conv is None:
                raise ValueError("ターゲットレイヤが見つかりませんでした。target_layer を明示指定してください。")
            return last_conv

    # ---------------------------
    # 可視化メソッド
    # ---------------------------
    def plot_random_samples(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_images: int = 16,
        cols: int = 4,
        save_dir: Optional[str] = None,
        cmap_title: bool = True,
    ):
        print(f"ランダムな{num_images}枚の画像でGrad-CAMを可視化します...")

        batch = next(iter(dataloader))
        images_tensor, labels = batch[0], batch[1]
        take = min(num_images, images_tensor.size(0))
        images_tensor = images_tensor[:take].to(self.device, non_blocking=True)
        labels = labels[:take]

        # ここは単なる予測なので no_grad でOK
        with torch.inference_mode():
            preds = self.model(images_tensor).argmax(dim=1)

        # ★ CAMは with で 1 回だけ生成 → ループで使い回し（__del__問題を回避）
        with GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=self.reshape_transform,
        ) as cam:
            visualized_images, titles = [], []
            for i in range(images_tensor.size(0)):
                pred_idx = int(preds[i].item())
                try:
                    visualization = self._make_cam_image_with(cam, images_tensor[i], pred_idx)
                except Exception as e:
                    h, w = images_tensor[i].shape[-2:]
                    visualization = np.ones((h, w, 3), dtype=np.float32)
                    print(f"[WARN] CAM generation failed at idx {i}: {e}")

                true_name = self._idx_to_name(int(labels[i]))
                pred_name = self._idx_to_name(pred_idx)
                title = f"True: {true_name}\nPred: {pred_name}"
                visualized_images.append(visualization)
                titles.append(title)

        self._plot_grid(
            visualized_images, titles, cols=cols,
            save_dir=save_dir, prefix="random", cmap_title=cmap_title
        )

    def plot_misclassified_samples(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_images: int = 16,
        cols: int = 4,
        save_dir: Optional[str] = None,
        cmap_title: bool = True,
    ):
        print(f"誤分類した画像を最大{max_images}枚探し、Grad-CAMを可視化します...")

        mis_images, titles = [], []
        found = 0

        # ★ ここも with で1回だけ生成
        with GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=self.reshape_transform,
        ) as cam:
            for images_tensor, labels in dataloader:
                if found >= max_images:
                    break

                images_tensor = images_tensor.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.inference_mode():
                    preds = self.model(images_tensor).argmax(dim=1)

                mis_idx = torch.where(preds != labels)[0]
                for idx in mis_idx[:max_images - found]:
                    pred_idx = int(preds[idx].item())
                    try:
                        visualization = self._make_cam_image_with(cam, images_tensor[idx], pred_idx)
                    except Exception as e:
                        h, w = images_tensor[idx].shape[-2:]
                        visualization = np.ones((h, w, 3), dtype=np.float32)
                        print(f"[WARN] CAM generation failed at mis idx {int(idx)}: {e}")

                    true_name = self._idx_to_name(int(labels[idx].item()))
                    pred_name = self._idx_to_name(pred_idx)
                    title = f"True: {true_name}\nPred: {pred_name}"
                    mis_images.append(visualization)
                    titles.append(title)
                    found += 1

        if not mis_images:
            print("誤分類された画像は見つかりませんでした。")
            return

        self._plot_grid(
            mis_images, titles, cols=cols,
            save_dir=save_dir, prefix="miscls", cmap_title=cmap_title
        )

    # ---------------------------
    # 内部ヘルパー
    # ---------------------------
    def _plot_grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        cols: int = 4,
        save_dir: Optional[str] = None,
        prefix: str = "cam",
        cmap_title: bool = True,
        save_individual: bool = False,
        grid_filename: Optional[str] = None,
        dpi: int = 200
    ):
        rows = (len(images) + cols - 1) // cols
        fig = plt.figure(figsize=(cols * 4, rows * 4))

        for i, (img, title) in enumerate(zip(images, titles)):
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            if cmap_title:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.tight_layout()

        # --- 保存ロジック ---
        out_dir = Path(save_dir) if save_dir is not None else (self.results_dir / prefix)
        out_dir.mkdir(parents=True, exist_ok=True)

        grid_name = grid_filename or f"{prefix}_grid.jpg"
        grid_path = out_dir / grid_name
        try:
            fig.savefig(grid_path, dpi=dpi, bbox_inches="tight")
            print(f"グリッド画像を保存しました: {grid_path}")
        except Exception as e:
            print(f"グリッド画像の保存に失敗: {e}")

        # 画面表示（任意）
        plt.show()
        plt.close(fig)

        if save_individual:
            for i, img in enumerate(images):
                tile_path = out_dir / f"{prefix}_{i:03d}.jpg"
                try:
                    plt.imsave(tile_path, img)
                except Exception as e:
                    print(f"個別保存に失敗({tile_path}): {e}")
            if images:
                print(f"個別画像を保存しました: {out_dir}")

        # W&Bログ：保存後にログ
        try:
            self.wand_run.log({f"{prefix}/grid_image": wandb.Image(str(grid_path), caption=f"{prefix} grid")})
        except Exception as e:
            print(f"W&Bログはスキップ: {e}")

    def _idx_to_name(self, idx: int) -> str:
        if self.class_names is None:
            return str(idx)
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)

    def _make_cam_image_with(self, cam: GradCAM, img_tensor: torch.Tensor, pred_idx: int) -> np.ndarray:
        """
        CAM計算は勾配が必要。no_grad/inference_modeで囲まない。
        ここでは with で受け取った cam インスタンスを使用して1枚分を可視化。
        """
        if img_tensor.dtype != torch.float32:
            img_tensor = img_tensor.float()
        input_tensor = img_tensor.unsqueeze(0)
        targets = [ClassifierOutputTarget(int(pred_idx))]

        # 勾配有効でCAMを計算（pytorch-grad-cam側でbackwardを使用）
        torch.set_grad_enabled(True)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        rgb_img = denormalize_image(img_tensor, self.mean, self.std)
        return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
