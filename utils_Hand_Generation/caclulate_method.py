import numpy as np
import torch
import torch.nn.functional as F
import random
from scipy.spatial import cKDTree

def z_rotation_matrix(angle_vec):
    """
    angle_vec: (B * sampling, 2) — (cosθ, sinθ)
    returns:   (B * sampling, 3, 3) z軸周りの回転行列
    """
    cos_theta = angle_vec[:, :, 0]
    sin_theta = angle_vec[:, :, 1]

    B, N, _ = angle_vec.shape 

    R = torch.zeros((B, N, 3, 3)).cuda()

    R[:, :, 0, 0] = cos_theta
    R[:, :, 0, 1] = -sin_theta
    R[:, :, 1, 0] = sin_theta
    R[:, :, 1, 1] = cos_theta
    R[:, :, 2, 2] = 1.0

    return R

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

def Loss_Grable(predhand, target_hand, partsc, D_threshold=0.1, batchSize=8):
    """
    prc: (B, 1, 3) - 基準座標
    target_hand_r: (B, 23, 3) - 指の座標を指先座標に変換
    D_threshold: float - 許容誤差倍率（例：0.1 → 10%以内はOK）
    """
    # 各指先ごとの距離 (B, 5)
    #partscenter = partsc.repeat(5,1)
    partscenter = partsc.cuda()
    penalties_sum = torch.zeros(batchSize).cuda()

    for i in range(5):
        j = i + 18
        pred_fingers = predhand[:, j]
        target_fingers = target_hand[: , j]
        diff_target = partscenter - target_fingers
        ref_dist = torch.norm(diff_target, dim=1)
        allowed_dist = ref_dist * (1 + D_threshold)

        diff = partscenter - pred_fingers
        dist = torch.norm(diff, dim=1) # ユークリッド距離
        penalties = F.mse_loss(dist , allowed_dist)# (B, 5)
        penalties_sum += penalties  

    return penalties_sum.unsqueeze(1)

def distance_margin_loss(p1, p2, D, margin=0.0):
    """
    p1, p2: Tensor of shape (..., 3)
    p1: partsの中心
    p2: 指先座標
    D: 閾値距離
    margin: 余裕を持たせるマージン（任意）
    """
    distance = torch.norm(p1 - p2, dim=-1)  # ユークリッド距離
    loss = F.relu(distance - D - margin)    # D以下ならlossは0、それより離れると徐々に増える
    return loss.mean()

def farthest_point_sampling(points, target_num=100):
    sampled = [random.randint(0, len(points) - 1)]
    dists = np.full(len(points), np.inf)
    for _ in range(1, target_num):
        last = points[sampled[-1]]
        dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))
        sampled.append(np.argmax(dists))
    return points[sampled]

#chatgpt
def calculate_accuracy(logits, labels, top_k=(1, 5)):
    """
    logits: 類似度スコアの行列 (N x N)
    labels: 正解のラベル (torch.arange(N))
    top_k: 求めるTop-k精度のリスト
    """
    max_k = max(top_k)
    batch_size = logits.size(0)

    # logitsからtop-kのインデックスを取得
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)

    # 正解のラベルを拡張して比較可能にする
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))

    # 各kに対して精度を計算
    accuracies = {}
    for k in top_k:
        correct_k = correct[:, :k].reshape(-1).float().sum(0, keepdim=True)
        accuracies[f'Top-{k}'] = correct_k.mul_(100.0 / batch_size).item()

    return accuracies

def calc_Dandz(wrist):
    D = wrist[:, 0]**2 + wrist[:, 1]**2
    z = wrist[:,2]
    return D, z

def rotation_matrix_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def calc_chamfer(p1, p2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    # p1 → p2 の最小距離
    dist1, _ = tree2.query(p1)
    # p2 → p1 の最小距離
    dist2, _ = tree1.query(p2)
    cd = np.mean(dist1 ** 2) + np.mean(dist2 ** 2)
    return cd

def decide_loss(p1):
    #p1をスケーリング
    #p1:  256 * 3
    pmove = np.expand_dims(np.mean(p1, axis = 0), 0)
    p_center = p1 - pmove # center
    dist = np.max(np.sqrt(np.sum(p_center ** 2, axis = 1)),0)
    p1 = (p1) / dist 
    cd = np.array([])
    for i in range(3):
        theta = np.pi * (i + 1) / 2
        R = rotation_matrix_z(theta)
        p2 = p1 @ R.T
        cdd = calc_chamfer(p1, p2)
        cd = np.append(cd, cdd)
    if cd.mean() <= 0.05:
        loss = Loss_Grable
    else:
        #対称性なし
        loss = F.mse_loss
    return loss, p1


# --- 各損失の平均値を計算（要素数が 0 の場合は None または 0 に）

import math

def get_z_rotation_matrix(angle_deg):
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    R = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=torch.float32)  # [3, 3]
    return R

def rotate_targets_z(target, N):
    """
    target: [B, 23, 3]
    returns: [B, N, 23, 3]
    """
    B, J, D = target.shape
    assert D == 3
    rotated = []

    for n in range(N):
        angle = 30 * n  # 0, 30, ..., 330
        R = get_z_rotation_matrix(angle).to(target.device)  # [3, 3]
        target_rotated = torch.matmul(target, R.T)  # [B, 23, 3]
        rotated.append(target_rotated.unsqueeze(1))  # → [B, 1, 23, 3]
    rotated_all = torch.cat(rotated, dim=1)  # [B, N, 23, 3]
    return rotated_all

def sekitori_loss_mean(pred, target):
    """
    pred:   [B, N, J, D]  # サンプリングされた予測リスト
    target: [B, T, J, D]  # 教師データ (回転したT個)

    各教師(T個)に対して、重複なくK個のpredを割り当てる。
    戻り値:
        losses:  torch.Tensor, [B, T, K] 割り当てられたloss (autograd対応)
        indices: np.ndarray,  [B, T, K] 割り当てられたpredのインデックス
        loss_mean_target: torch.Tensor, [B, T] 各ターゲットの平均loss
        indices_min_loss: list of np.ndarray, 各バッチのpredごとに最小lossの教師index
    """
    B, N, J, D = pred.shape
    T = target.shape[1]
    assert N % T == 0, "predの数が教師で割り切れません"
    K = N // T

    all_losses = []
    all_indices = []
    indices_min_loss = []

    for b in range(B):
        # 1. loss_matrix [N, T] を作成
        loss_matrix = torch.zeros(N, T, device=pred.device)
        for i in range(N):
            for j in range(T):
                loss_matrix[i, j] = F.mse_loss(pred[b, i], target[b, j], reduction='mean')

        # 2. predごとに最小の教師を記録（どの教師に近いか）
        min_loss, min_loss_idx = torch.min(loss_matrix, dim=1)
        indices_min_loss.append(min_loss_idx.detach().cpu().numpy())

        # 3. predを各教師に割り当てる
        sorted_gst = torch.argsort(min_loss)  # 最小loss順にpredを処理
        sel_list = [[] for _ in range(T)] # 各教師に割り当てられたlossを格納
        sel_indices_list = [[] for _ in range(T)]  # index
        counts = torch.zeros(T, dtype=torch.long)

        for g in sorted_gst:
            prefs = torch.argsort(loss_matrix[g]) # このpredに対する教師の優先順位
            for t in prefs:
                ti = t.item()
                if counts[ti] < K:
                    sel_list[ti].append(loss_matrix[g, t])  # 勾配追跡可能
                    sel_indices_list[ti].append(g.item())   # pred index
                    counts[ti] += 1
                    break
            if all(c == K for c in counts): # 全員割り当て終わったら終了
                break

        # 4. Tensorに変換
        chosen_losses = torch.stack([torch.stack(v) for v in sel_list])      # [T, K]
        chosen_indices = np.array(sel_indices_list)                          # [T, K]

        all_losses.append(chosen_losses)
        all_indices.append(chosen_indices)

    # 5. バッチ方向にstack
    losses_tensor = torch.stack(all_losses, dim=0)     # [B, T, K]
    indices_use_loss = np.stack(all_indices, axis=0)   # [B, T, K]
    indices_loss_min = np.stack(indices_min_loss, axis=0) # list of length B * N 

    # 6. 各教師の平均lossと最悪ターゲット
    loss_mean_target = torch.mean(losses_tensor, dim=2)    # [B, T] ターゲットごとの平均値
    losses_worst, worst_idx = torch.max(loss_mean_target, dim=1)  # [B] 最悪の教師
    losses_mean = torch.mean(loss_mean_target)  # [B] 各ターゲットの平均

    return losses_mean, worst_idx.detach().cpu().numpy(), loss_mean_target.detach().cpu().numpy(), indices_loss_min, indices_use_loss

def sekitori_loss_sum(pred, target):
    """
    pred:   [B, N, J, D]  # サンプリングされた予測リスト
    target: [B, T, J, D]  # 教師データ (回転したT個)

    各教師(T個)に対して、重複なくK個のpredを割り当てる。
    戻り値:
        losses:  torch.Tensor, [B, T, K] 割り当てられたloss (autograd対応)
        indices: np.ndarray,  [B, T, K] 割り当てられたpredのインデックス
        loss_mean_target: torch.Tensor, [B, T] 各ターゲットの平均loss
        indices_min_loss: list of np.ndarray, 各バッチのpredごとに最小lossの教師index
    """
    B, N, J, D = pred.shape
    T = target.shape[1]
    assert N % T == 0, "predの数が教師で割り切れません"
    K = N // T

    all_losses = []
    all_indices = []
    indices_min_loss = []

    for b in range(B):
        # 1. loss_matrix [N, T] を作成
        loss_matrix = torch.zeros(N, T, device=pred.device)
        for i in range(N):
            for j in range(T):
                loss_matrix[i, j] = F.mse_loss(pred[b, i], target[b, j], reduction='sum') 

        # 2. predごとに最小の教師を記録（どの教師に近いか） 
        min_loss, min_loss_idx = torch.min(loss_matrix, dim=1)
        indices_min_loss.append(min_loss_idx.detach().cpu().numpy())

        # 3. predを各教師に割り当てる
        sorted_gst = torch.argsort(min_loss)  # 最小loss順にpredを処理
        sel_list = [[] for _ in range(T)] # 各教師に割り当てられたlossを格納
        sel_indices_list = [[] for _ in range(T)]  # index
        counts = torch.zeros(T, dtype=torch.long)

        for g in sorted_gst:
            prefs = torch.argsort(loss_matrix[g]) # このpredに対する教師の優先順位
            for t in prefs:
                ti = t.item()
                if counts[ti] < K:
                    sel_list[ti].append(loss_matrix[g, t])  # 勾配追跡可能
                    sel_indices_list[ti].append(g.item())   # pred index
                    counts[ti] += 1
                    break
            if torch.all(counts == K): # 全員割り当て終わったら終了
                break

        # 4. Tensorに変換
        chosen_losses = torch.stack([torch.stack(v) for v in sel_list])      # [T, K]
        chosen_indices = np.array(sel_indices_list)                          # [T, K]

        all_losses.append(chosen_losses) 
        all_indices.append(chosen_indices)

    # 5. バッチ方向にstack
    losses_tensor = torch.stack(all_losses, dim=0)     # [B, T, K]
    indices_use_loss = np.stack(all_indices, axis=0)   # [B, T, K]
    indices_loss_min = np.stack(indices_min_loss, axis=0) # list of length B * N 

    # 6. 各教師の平均lossと最悪ターゲット
    loss_mean_target = torch.mean(losses_tensor, dim=2)    # [B, T] ターゲットごとの平均値 # dim 2?
    losses_worst, worst_idx = torch.max(loss_mean_target, dim=1)  # [B] 最悪の教師
    losses_mean = torch.mean(loss_mean_target)  # [B] 各ターゲットの平均
    # meanではなくsumの可能性あり
    return losses_mean, worst_idx.detach().cpu().numpy(), loss_mean_target.detach().cpu().numpy() / 69, indices_loss_min, indices_use_loss

def sekitori_loss_worst2(pred, target, worst_percent):
    """
    pred:   [B, N, J, D]  # サンプリングされた予測リスト
    target: [B, T, J, D]  # 教師データ (回転したT個)

    各教師(T個)に対して、重複なくK個のpredを割り当てる。
    戻り値:
        losses:  torch.Tensor, [B, T, K] 割り当てられたloss (autograd対応)
        indices: np.ndarray,  [B, T, K] 割り当てられたpredのインデックス
        loss_mean_target: torch.Tensor, [B, T] 各ターゲットの平均loss
        indices_min_loss: list of np.ndarray, 各バッチのpredごとに最小lossの教師index
    """
    B, N, J, D = pred.shape
    T = target.shape[1]
    assert N % T == 0, "predの数が教師で割り切れません"
    K = N // T

    all_losses = []
    all_indices = []
    indices_min_loss = []

    for b in range(B):
        # 1. loss_matrix [N, T] を作成
        loss_matrix = torch.zeros(N, T, device=pred.device)
        for i in range(N):
            for j in range(T):
                loss_matrix[i, j] = F.mse_loss(pred[b, i], target[b, j], reduction='sum') 

        # 2. predごとに最小の教師を記録（どの教師に近いか）
        min_loss, min_loss_idx = torch.min(loss_matrix, dim=1)
        indices_min_loss.append(min_loss_idx.detach().cpu().numpy())

        # 3. predを各教師に割り当てる
        sorted_gst = torch.argsort(min_loss)  # 最小loss順にpredを処理
        sel_list = [[] for _ in range(T)] # 各教師に割り当てられたlossを格納
        sel_indices_list = [[] for _ in range(T)]  # index
        counts = torch.zeros(T, dtype=torch.long)

        for g in sorted_gst:
            prefs = torch.argsort(loss_matrix[g]) # このpredに対する教師の優先順位
            for t in prefs:
                ti = t.item()
                if counts[ti] < K:
                    sel_list[ti].append(loss_matrix[g, t])  # 勾配追跡可能
                    sel_indices_list[ti].append(g.item())   # pred index
                    counts[ti] += 1
                    break
            if all(c == K for c in counts): # 全員割り当て終わったら終了
                break

        # 4. Tensorに変換
        chosen_losses = torch.stack([torch.stack(v) for v in sel_list])      # [T, K]
        chosen_indices = np.array(sel_indices_list)                          # [T, K]

        all_losses.append(chosen_losses) 
        all_indices.append(chosen_indices)

    # 5. バッチ方向にstack
    losses_tensor = torch.stack(all_losses, dim=0)     # [B, T, K]
    indices_use_loss = np.stack(all_indices, axis=0)   # [B, T, K]
    indices_loss_min = np.stack(indices_min_loss, axis=0) # list of length B * N 

    # 6. 各教師の平均lossと最悪ターゲット
    loss_mean_target = torch.mean(losses_tensor, dim=2)    # [B, T] ターゲットごとの平均値 # dim 2?
    _, worst_idx = torch.max(loss_mean_target, dim=1)  # [B] 最悪の教師
    use_worst_num = int(N * worst_percent / 100)  #下位何パーセントのLossでbackwardするか？
    losses_worst, _ = torch.topk(losses_tensor.reshape(1, -1), use_worst_num)  #  席取り後のworst30
    loss_worst = torch.mean(losses_worst)
    sorted_flat = torch.sort(losses_tensor.reshape(1, -1), dim=1, descending=True)[0]
    losses_mean = torch.mean(loss_mean_target)  # [B] 各ターゲットの平均
    return loss_worst, worst_idx.detach().cpu().numpy(), loss_mean_target.detach().cpu().numpy() / 69, indices_loss_min, indices_use_loss, sorted_flat.detach().cpu().numpy() / 69

def sekitori_loss_worst(pred, target):
    """
    pred:   [B, N, J, D]  # サンプリングされた予測リスト
    target: [B, T, J, D]  # 教師データ (回転したT個)
    各教師(T個)に対して、重複なくK個のpredを割り当てる。
    戻り値:
        losses:  torch.Tensor, [B, T, K] 割り当てられたloss (autograd対応)
        indices: np.ndarray,  [B, T, K] 割り当てられたpredのインデックス
        loss_mean_target: torch.Tensor, [B, T] 各ターゲットの平均loss
        indices_min_loss: list of np.ndarray, 各バッチのpredごとに最小lossの教師index
    """
    B, N, J, D = pred.shape
    T = target.shape[1]
    assert N % T == 0, "predの数が教師で割り切れません"
    K = N // T

    all_losses = []
    all_indices = []
    indices_min_loss = []

    for b in range(B):
        # 1. loss_matrix [N, T] を作成
        loss_matrix = torch.zeros(N, T, device=pred.device)
        for i in range(N):
            for j in range(T):
                loss_matrix[i, j] = F.mse_loss(pred[b, i], target[b, j], reduction='sum') 

        # 2. predごとに最小の教師を記録（どの教師に近いか）
        min_loss, min_loss_idx = torch.min(loss_matrix, dim=1)
        indices_min_loss.append(min_loss_idx.detach().cpu().numpy())

        # 3. predを各教師に割り当てる
        sorted_gst = torch.argsort(min_loss)  # 最小loss順にpredを処理
        sel_list = [[] for _ in range(T)] # 各教師に割り当てられたlossを格納
        sel_indices_list = [[] for _ in range(T)]  # index
        counts = torch.zeros(T, dtype=torch.long)

        for g in sorted_gst:
            prefs = torch.argsort(loss_matrix[g]) # このpredに対する教師の優先順位
            for t in prefs:
                ti = t.item()
                if counts[ti] < K:
                    sel_list[ti].append(loss_matrix[g, t])  # 勾配追跡可能
                    sel_indices_list[ti].append(g.item())   # pred index
                    counts[ti] += 1
                    break
            if all(c == K for c in counts): # 全員割り当て終わったら終了
                break

        # 4. Tensorに変換
        chosen_losses = torch.stack([torch.stack(v) for v in sel_list])      # [T, K]
        chosen_indices = np.array(sel_indices_list)                          # [T, K]

        all_losses.append(chosen_losses) 
        all_indices.append(chosen_indices)

    # 5. バッチ方向にstack
    losses_tensor = torch.stack(all_losses, dim=0)     # [B, T, K]
    indices_use_loss = np.stack(all_indices, axis=0)   # [B, T, K]
    indices_loss_min = np.stack(indices_min_loss, axis=0) # list of length B * N 

    # 6. 各教師の平均lossと最悪ターゲット
    loss_mean_target = torch.mean(losses_tensor, dim=2)    # [B, T] ターゲットごとの平均値 # dim 2?
    losses_worst, worst_idx = torch.max(loss_mean_target, dim=1)  # [B] 最悪の教師
    losses_mean = torch.mean(loss_mean_target)  # [B] 各ターゲットの平均
    
    return losses_worst[0], worst_idx.detach().cpu().numpy(), loss_mean_target.detach().cpu().numpy() / 69, indices_loss_min, indices_use_loss

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