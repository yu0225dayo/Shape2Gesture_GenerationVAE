import numpy as np
import torch
import torch.nn.functional as F
import random
from scipy.spatial import cKDTree

def chamfer_distance(p1):
    """
    p1: (B, N, D) 点群1
    p2: (B, M, D) 点群2
    """
    theta = torch.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    R = torch.tensor([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]])
    theta, cos, sin, R = theta.cuda(), cos.cuda(), sin.cuda(), R.cuda()
    p2 = p1 @ R.T
    # 距離行列 (B, N, M)
    dist = torch.cdist(p1, p2, p=2)
    # p1 → p2 の最小距離 (B, N)
    min_dist_p1_p2 = dist.min(dim=2)[0]
    # p2 → p1 の最小距離 (B, M)
    min_dist_p2_p1 = dist.min(dim=1)[0]
    # 平均して Chamfer 距離を返す
    loss = min_dist_p1_p2.mean(dim=1) + min_dist_p2_p1.mean(dim=1)
    return loss.mean()

def calc_chamfer(p1, p2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    # p1 → p2 の最小距離
    dist1, _ = tree2.query(p1)
    # p2 → p1 の最小距離
    dist2, _ = tree1.query(p2)
    cd = np.mean(dist1 ** 2) + np.mean(dist2 ** 2)
    return cd

def farthest_point_sampling(points, target_num=100):
    sampled = [random.randint(0, len(points) - 1)]
    dists = np.full(len(points), np.inf)
    for _ in range(1, target_num):
        last = points[sampled[-1]]
        dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))
        sampled.append(np.argmax(dists))
    return points[sampled]

def get_patseg(model, point, target, num_classes=3):
    """
    input:
        model: PointNetDenseCls
        data: (points, target, hand_target, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist_target)
    returns:
        pl, pr        : centering + scaling されたパーツ点群 (B,3,256)
        all_feat      : 全体形状特徴ベクトル
        plout, prout  : scaling のみで centering しないパーツ点群 (B,3,256)
    """
    B = point.size(0)
    points = point.transpose(2, 1)  # (B,3,N)
    partsseg_target = target
    points, target = points.cuda(), target.cuda()

    # parts segmentation
    pred, trans, trans_feat, all_feat = model(points)
    pred = pred.view(-1, num_classes)
    target = target.view(-1, 1)[:, 0]
    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()

    # pred_choice 1 : left,  pred_choice 2 : right
    points = points.transpose(1, 2).cpu().data.numpy()  # (B,2048,3)
    pred_np = pred_np.reshape(B, 2048, 1)

    pl_out, pr_out = np.empty((0, 3)), np.empty((0, 3))   # centering+scaling
    plout_raw, prout_raw = np.empty((0, 3)), np.empty((0, 3))  # scalingのみ
    plc, prc = np.empty((0, 3)), np.empty((0, 3))

    for batch in range(B):
        parts_l_list = np.array([])
        parts_r_list = np.array([])

        # ラベルが明らかに少なければ教師をそのまま利用
        if np.count_nonzero(pred_np[batch] == 2) <= 10:
            target_l = partsseg_target
        else:
            target_l = pred_np
        if np.count_nonzero(pred_np[batch] == 1) <= 10:
            target_r = partsseg_target
        else:
            target_r = pred_np

        for j in range(2048):
            if target_l[batch][j] == 2:
                parts_l_list = np.append(parts_l_list, points[batch][j])
            if target_r[batch][j] == 1:
                parts_r_list = np.append(parts_r_list, points[batch][j])

        # パーツ点が少なければコピー拡張
        while len(parts_l_list) <= (3 * 256):
            add_list = parts_l_list * 1.01
            parts_l_list = np.append(parts_l_list, add_list)
        while len(parts_r_list) <= (3 * 256):
            add_list = parts_r_list * 1.01
            parts_r_list = np.append(parts_r_list, add_list)

        # FPS sampling
        parts_l_list = parts_l_list.reshape(int(len(parts_l_list) / 3), 3)
        pl = farthest_point_sampling(parts_l_list, target_num=256)
        parts_r_list = parts_r_list.reshape(int(len(parts_r_list) / 3), 3)
        pr = farthest_point_sampling(parts_r_list, target_num=256)

        plout_raw = np.vstack((plout_raw, pl))
        prout_raw = np.vstack((prout_raw, pr))

        # === scaling + centering ===
        pl_center = np.expand_dims(np.mean(pl, axis=0), 0)
        pl_c = (pl - pl_center)
        dist_l = np.max(np.sqrt(np.sum(pl_c ** 2, axis=1)), 0)

        pr_center = np.expand_dims(np.mean(pr, axis=0), 0)
        pr_c = (pr - pr_center)
        dist_r = np.max(np.sqrt(np.sum(pr_c ** 2, axis=1)), 0)

        pl_out = np.vstack((pl_out, pl_c / dist_l))
        pr_out = np.vstack((pr_out, pr_c / dist_r))
        plc = np.vstack((plc, pl_center))
        prc = np.vstack((prc, pr_center))
        # === scalingのみ (centeringなし) ===

    # numpy → torch (centering+scaling)
    pl = torch.from_numpy(pl_out.reshape(B, 256, 3).astype(np.float32)).cuda().transpose(2, 1)
    pr = torch.from_numpy(pr_out.reshape(B, 256, 3).astype(np.float32)).cuda().transpose(2, 1)

    # numpy → torch 
    plout = torch.from_numpy(plout_raw.reshape(B, 256, 3).astype(np.float32)).cuda().transpose(2, 1)
    prout = torch.from_numpy(prout_raw.reshape(B, 256, 3).astype(np.float32)).cuda().transpose(2, 1)

    return pl, pr, all_feat, plout, prout

def get_patseg_target(model, point, target):
    """
    input:
        model: PointNetDenseCls
        data: (points, target, hand_target, filename, batch_weight, hand_set, hand_scale, hand_format, sita_ans, wrist_target)
    returns:
        pl, pr        : centering + scaling されたパーツ点群 (B,3,256)
        all_feat      : 全体形状特徴ベクトル
        plout, prout  : scaling のみで centering しないパーツ点群 (B,3,256)
    """
    B = point.size(0)
    points = point.transpose(2, 1)  # (B,3,N)
    partsseg_target = target

    points, target = points.cuda(), target.cuda()

    # parts segmentation
    pred, trans, trans_feat, all_feat = model(points)
    pred = pred.view(-1, 3)
    target = target.view(-1, 1)[:, 0]
    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()

    # pred_choice 1 : left,  pred_choice 2 : right
    points = points.transpose(1, 2).cpu().data.numpy()  # (B,2048,3)
    pred_np = pred_np.reshape(B, 2048, 1)

    pl_out, pr_out = np.empty((0, 3)), np.empty((0, 3))   # centering+scaling
    plout_raw, prout_raw = np.empty((0, 3)), np.empty((0, 3))  # scalingのみ
    plc, prc = np.empty((0, 3)), np.empty((0, 3))

    for batch in range(B):
        parts_l_list = np.array([])
        parts_r_list = np.array([])
        target_l = partsseg_target
        target_r = partsseg_target

        for j in range(2048):
            if target_l[batch][j] == 2:
                parts_l_list = np.append(parts_l_list, points[batch][j])
            if target_r[batch][j] == 1:
                parts_r_list = np.append(parts_r_list, points[batch][j])

        # パーツ点が少なければコピー拡張
        while len(parts_l_list) <= (3 * 256):
            add_list = parts_l_list * 1.01
            parts_l_list = np.append(parts_l_list, add_list)
        while len(parts_r_list) <= (3 * 256):
            add_list = parts_r_list * 1.01
            parts_r_list = np.append(parts_r_list, add_list)

        # FPS sampling
        parts_l_list = parts_l_list.reshape(int(len(parts_l_list) / 3), 3)
        pl = farthest_point_sampling(parts_l_list, target_num=256)
        parts_r_list = parts_r_list.reshape(int(len(parts_r_list) / 3), 3)
        pr = farthest_point_sampling(parts_r_list, target_num=256)

        plout_raw = np.vstack((plout_raw, pl))
        prout_raw = np.vstack((prout_raw, pr))

        # === scaling + centering ===
        pl_center = np.expand_dims(np.mean(pl, axis=0), 0)
        pl_c = (pl - pl_center)
        dist_l = np.max(np.sqrt(np.sum(pl_c ** 2, axis=1)), 0)

        pr_center = np.expand_dims(np.mean(pr, axis=0), 0)
        pr_c = (pr - pr_center)
        dist_r = np.max(np.sqrt(np.sum(pr_c ** 2, axis=1)), 0)

        pl_out = np.vstack((pl_out, pl_c / dist_l))
        pr_out = np.vstack((pr_out, pr_c / dist_r))
        plc = np.vstack((plc, pl_center))
        prc = np.vstack((prc, pr_center))
        # === scalingのみ (centeringなし) ===

    # numpy → torch (centering+scaling)
    pl = torch.from_numpy(pl_out.reshape(B, 256, 3).astype(np.float32)).cuda().transpose(2, 1)
    pr = torch.from_numpy(pr_out.reshape(B, 256, 3).astype(np.float32)).cuda().transpose(2, 1)

    # numpy → torch 
    plout = torch.from_numpy(plout_raw.reshape(B, 256, 3).astype(np.float32)).cuda().transpose(2, 1)
    prout = torch.from_numpy(prout_raw.reshape(B, 256, 3).astype(np.float32)).cuda().transpose(2, 1)

    return pl, pr, all_feat, plout, prout