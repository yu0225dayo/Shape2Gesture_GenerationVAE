from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

from dataset_format_xy import *
from model_pointnet import PointNetDenseCls
from model import VAE, PartsEncoder_w_TNet, Position_Generater_VAE
from functions_pointnet_demo import get_patseg_target

# Mapping is fixed so colors stay the same across runs
CLASS_CODES = ["ba", "bo", "ju", "ka", "mu", "pa", "pc", "po", "va"]
CLASS_NAMES = {
    "ba": "basket",
    "bo": "bottle",
    "ju": "jug",
    "ka": "kettle",
    "mu": "mug",
    "pa": "pan",
    "pc": "pc",
    "po": "pot",
    "va": "vase",
}
CLASS_COLORS = {
    "ba": "#1f77b4",
    "bo": "#ff7f0e",
    "ju": "#2ca02c",
    "ka": "#d62728",
    "mu": "#8c564b",
    "pa": "#e377c2",
    "pc": "#7f7f7f",
    "po": "#bcbd22",
    "va": "#10bedd",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_CODES)}

def plot_ellipse_from_cov(ax, mu, cov_2x2, color, alpha=0.5, n_std=1.0):
    """
    2x2の共分散行列から楕円を描画
    """
    # 固有値・固有ベクトルを計算
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
    
    # 楕円の角度（ラジアン→度）
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # 楕円の幅と高さ（標準偏差×n_std×2）
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 1e-10))
    
    ell = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        linewidth=1.2,
        alpha=alpha,
    )
    ax.add_patch(ell)

def run_pca(mu_all, logvar_all, labels, side_label, out_dir, selected_codes):
    labels = np.array(labels)
    mask = np.isin(labels, selected_codes)
    if mask.sum() == 0:
        print(f"[WARN] no samples for {side_label} in selected labels: {selected_codes}")
        return

    mu_all = mu_all[mask]
    logvar_all = logvar_all[mask]
    labels = labels[mask]

    # PCAでmuを2次元に変換
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(mu_all)
    
    # PCA射影行列を取得
    W = pca.components_  # shape: (2, n_dims)
    
    # 各サンプルのlogvarをPCA空間の2x2共分散行列に変換
    logvar_2d = []
    for lv in logvar_all:
        # 元の空間での対角共分散行列（logvar -> variance）
        cov = np.diag(np.exp(lv))
        # PCA空間での共分散: W @ Cov @ W^T
        cov_2d = W @ cov @ W.T
        logvar_2d.append(cov_2d)
    
    logvar_2d = np.array(logvar_2d)

    fig, ax = plt.subplots(figsize=(8, 6))
    encoded_labels = [CLASS_TO_IDX[c] for c in labels]
    scatter = ax.scatter(
        mu_2d[:, 0],
        mu_2d[:, 1],
        c=encoded_labels,
        cmap=plt.matplotlib.colors.ListedColormap([CLASS_COLORS[c] for c in CLASS_CODES]),
        vmin=0,
        vmax=len(CLASS_CODES) - 1,
        s=18,
    )

    # 各クラスから最大3サンプルの楕円を描画
    for code in selected_codes:
        cls_mask = labels == code
        if not np.any(cls_mask):
            continue
        color = CLASS_COLORS[code]
        # 変換後の2x2共分散行列を使用
        for m, cov_2d in zip(mu_2d[cls_mask][:3], logvar_2d[cls_mask][:3]):
            plot_ellipse_from_cov(ax, m, cov_2d, color=color)

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f"{side_label} hand latent space (selected classes)")
    legend_handles = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in selected_codes if c in CLASS_COLORS]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Class", bbox_to_anchor=(1.02, 0.5), loc="center left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"pca_{side_label.lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")

def collect_latent(dataloader, pointnet, position_generater_l, position_generater_r):
    all_mu_l, all_logvar_l = [], []
    all_mu_r, all_logvar_r = [], []
    all_labels = []

    pointnet.eval()
    position_generater_l.eval()
    position_generater_r.eval()

    with torch.no_grad():
        for data in dataloader:
            points, target, hand_target, filename, *_ = data

            #pl, pr, all_feat, plout, prout = get_patseg(pointnet, points, target)
            pl, pr, all_feat, plout, prout = get_patseg_target(pointnet, points, target)

            mu_l, logvar_l = position_generater_l.get_mu_sigma(plout, all_feat)
            mu_r, logvar_r = position_generater_r.get_mu_sigma(prout, all_feat)

            batch_size = mu_l.shape[0]
            for b in range(batch_size):
                label_code = filename[b][:2]
                all_mu_l.append(mu_l[b].detach().cpu().numpy())
                all_logvar_l.append(logvar_l[b].detach().cpu().numpy())
                all_mu_r.append(mu_r[b].detach().cpu().numpy())
                all_logvar_r.append(logvar_r[b].detach().cpu().numpy())
                all_labels.append(label_code)

    return (
        np.stack(all_mu_l),
        np.stack(all_logvar_l),
        np.stack(all_mu_r),
        np.stack(all_logvar_r),
        np.array(all_labels),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
    parser.add_argument("--workers", type=int, default=0, help="number of data loading workers")
    parser.add_argument("--dataset", type=str, default="neuralnet_dataset_unity", help="dataset path")
    parser.add_argument("--select_labels", type=list, default=["ba", "bo", "ju", "ka", "mu", "pa", "pc", "po", "va"], help="class codes to include (e.g. ju mu bo pc)")
    parser.add_argument("--outf", type=str, default="PCA_selected", help="output folder for PCA plots")
    opt = parser.parse_args()

    opt.select_labels = [c for c in opt.select_labels if c in CLASS_CODES]
    if len(opt.select_labels) == 0:
        raise ValueError(f"No valid select_labels given. Valid codes: {CLASS_CODES}")


    dataset = ShapeNetDataset_format_select(
        root=opt.dataset,
        data_augmentation=False,
        select_labels=opt.select_labels,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers),
    )

    pointnet = PointNetDenseCls(k=3, feature_transform=False)
    state_dict_pointnet = torch.load("save_pretrained_partsseg/pointnet_model_sotuken_acc_partseg_best.pth", weights_only=True)
    pointnet.load_state_dict(state_dict_pointnet)
    pointnet.cuda()

    parts_encoder_l, parts_encoder_r = PartsEncoder_w_TNet(), PartsEncoder_w_TNet()
    state_parts_e_l = torch.load("fps_formatxy2/parts_encoder_l_model_sotuken_loss_mse_best.pth", weights_only=True)
    state_parts_e_r = torch.load("fps_formatxy2/parts_encoder_r_model_sotuken_loss_mse_best.pth", weights_only=True)
    parts_encoder_l.load_state_dict(state_parts_e_l)
    parts_encoder_r.load_state_dict(state_parts_e_r)
    parts_encoder_l.eval().cuda()
    parts_encoder_r.eval().cuda()

    position_generater_l, position_generater_r = Position_Generater_VAE(), Position_Generater_VAE()
    state_sita_l = torch.load("worst30_sampler/position_generater_l/Epoch/29_epoch.pth", weights_only=True)
    state_sita_r = torch.load("worst30_sampler/position_generater_r/Epoch/29_epoch.pth", weights_only=True)
    position_generater_l.load_state_dict(state_sita_l)
    position_generater_r.load_state_dict(state_sita_r)
    position_generater_l.eval().cuda()
    position_generater_r.eval().cuda()

    print(f"Selected labels: {opt.select_labels}")
    (
        all_mu_l,
        all_logvar_l,
        all_mu_r,
        all_logvar_r,
        all_labels,
    ) = collect_latent(dataloader, pointnet, position_generater_l, position_generater_r)

    out_dir = os.path.join(opt.outf, "_".join(opt.select_labels))
    run_pca(all_mu_l, all_logvar_l, all_labels, "Left", out_dir, opt.select_labels)
    run_pca(all_mu_r, all_logvar_r, all_labels, "Right", out_dir, opt.select_labels)

