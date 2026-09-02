import torch
import torch.nn.functional as F
from torch import nn


def contrastive_loss(z_i, z_j, temperature):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.

    Args:
        z_i: Projected features from view i, shape [B, D]
        z_j: Projected features from view j, shape [B, D]
        temperature: Temperature scaling parameter

    Returns:
        Scalar contrastive loss
    """
    batch_size = z_i.shape[0]
    out = torch.cat([z_i, z_j], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def contrastive_loss_with_graph(z_i, z_j, temperature, w_i, w_j, y):
    """
    Args:
        z_i, z_j: [B, D] 两个视图的特征
        temperature: 温度
        w_i: [B, B] 视图 i 的 K 近邻图，1=近邻
        w_j: [B, B] 视图 j 的 K 近邻图，1=近邻
        y: [B, B] 伪标签一致性图，1=同簇
    Returns:
        scalar loss
    """
    B = z_i.shape[0]
    device = z_i.device

    # ---- 1. 构建 logits 矩阵 ----
    out = torch.cat([z_i, z_j], dim=0)               # [2B, D]
    logits = torch.mm(out, out.t()) / temperature    # [2B, 2B]

    # ---- 2. 构建正样本掩码（2B×2B） ----
    # 2.1 原始跨视图自身正样本：无条件为 1
    pos_mask = torch.zeros(2*B, 2*B, device=device)
    pos_mask[torch.arange(B), torch.arange(B)+B] = 1.0   # z_i -> 对应 z_j
    pos_mask[torch.arange(B)+B, torch.arange(B)] = 1.0   # z_j -> 对应 z_i

    # 2.2 基于各视图近邻图与伪标签的附加正样本关系（排除自身）
    # 视图 i 的附加正样本关系矩阵 (B x B)，不含对角线
    pos_i = (w_i * y).float()
    pos_i_off = pos_i * (1 - torch.eye(B, device=device))

    # 视图 j 的附加正样本关系矩阵
    pos_j = (w_j * y).float()
    pos_j_off = pos_j * (1 - torch.eye(B, device=device))

    # 2.3 同视图内部正样本：z_i 内部，z_j 内部
    pos_mask[:B, :B] = torch.max(pos_mask[:B, :B], pos_i_off)
    pos_mask[B:, B:] = torch.max(pos_mask[B:, B:], pos_j_off)

    # 2.4 跨视图额外正样本
    # 对于视图 i 的 anchor（前 B 行），跨视图正样本由 pos_i_off 决定
    pos_mask[:B, B:] = torch.max(pos_mask[:B, B:], pos_i_off)
    # 对于视图 j 的 anchor（后 B 行），跨视图正样本由 pos_j_off 决定
    pos_mask[B:, :B] = torch.max(pos_mask[B:, :B], pos_j_off)

    # 3. 自身掩码：仅同视图对角线需从分母排除
    self_mask = torch.eye(2 * B, device=device)  # 对角线为 1
    # exclude_mask = (self_mask + pos_mask).clamp(max=1.0)

    # 4. 分母：除自身外的所有样本（含正样本，保持 InfoNCE 形式）
    denom_mask = (1.0 - self_mask).bool()
    # denom_mask = (1.0 - exclude_mask).bool()

    # 5. 分子：仅正样本
    pos_mask_bool = pos_mask.bool()

    # 6. 计算损失（对数空间）
    denom_logits = logits.masked_fill(~denom_mask, float('-inf'))
    denom_logsum = denom_logits.logsumexp(dim=1)  # [2B]

    pos_logits = logits.masked_fill(~pos_mask_bool, float('-inf'))
    pos_logsum = pos_logits.logsumexp(dim=1)  # [2B]

    # 跳过无正样本的 anchor（理论上一定有，因为至少有一个原始跨视图自身正样本）
    has_pos = pos_mask.sum(dim=1) > 0
    loss_per_sample = denom_logsum - pos_logsum  # -log(Σ_pos / Σ_denom)
    loss = loss_per_sample[has_pos].mean()
    return loss


class ClusterContrastiveLoss(nn.Module):
    """Cluster-level contrastive loss for view alignment."""

    def __init__(self, num_clusters, temperature_l):
        super(ClusterContrastiveLoss, self).__init__()
        self.num_clusters = num_clusters
        self.temperature_l = temperature_l

    def compute_cluster_loss(self, q_centers, k_centers, psedo_labels):
        """
        q_centers: 共识聚类中心 [num_clusters, feature_dim]
        k_centers: 视图特定的聚类中心 [num_clusters, feature_dim]
        psedo_labels: 当前batch的伪标签 [batch_size]
        """
        d_q = q_centers.mm(q_centers.T) / self.temperature_l
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        d_q = d_q.float()
        d_q[torch.arange(self.num_clusters), torch.arange(self.num_clusters)] = d_k

        device = q_centers.device
        zero_classes = torch.arange(self.num_clusters, device=device)[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                                          self.num_clusters),
                                                                                dim=0) == 0]
        mask = torch.zeros((self.num_clusters, self.num_clusters), dtype=torch.bool, device=device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)

        pos = d_q.diag(0)
        mask = torch.ones((self.num_clusters, self.num_clusters), device=device)
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.num_clusters - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.num_clusters, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.num_clusters - len(zero_classes))

        return loss


