import torch
import torch.nn.functional as F


class KMeans:
    """Optimized K-Means clustering in PyTorch."""

    def __init__(self, n_clusters=10, max_iter=None,
                 verbose=True, device=torch.device("cpu"), tol=1e-3):
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None          # (n_samples, n_clusters)
        self.centers = None
        self.variation = torch.tensor(float("inf"))
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.count = 0
        self.device = device
        self.started = False

    def fit(self, x):
        """Fit K-Means and return cluster assignments."""
        x = x.to(self.device)
        n_samples = x.shape[0]

        # 随机初始化中心
        init_idx = torch.randint(0, n_samples, (self.n_clusters,), device=self.device)
        self.centers = x[init_idx].clone()
        self.variation = torch.tensor(float("inf"), device=self.device)
        self.started = False
        self.count = 0

        while True:
            self.nearest_center(x)
            self.update_center(x)

            if self.verbose:
                print(self.variation.item(), torch.argmin(self.dists, dim=1))

            # 收敛判断
            if self.started and torch.abs(self.variation) < self.tol and self.max_iter is None:
                break
            if self.max_iter is not None and self.count >= self.max_iter:
                break

            self.count += 1

        return self.get_assignments()

    def nearest_center(self, x):
        """Assign each sample to the nearest center – fully vectorized."""
        # 使用 cdist 一次性计算所有样本到所有中心的欧氏距离，再平方
        dist = torch.cdist(x, self.centers) ** 2   # (N, K)
        labels = torch.argmin(dist, dim=1)         # (N,)

        if self.started:
            self.variation = torch.sum(self.dists - dist)
        self.dists = dist
        self.labels = labels
        self.started = True

    def update_center(self, x):
        """Update cluster centers via weighted average – fully vectorized."""
        # one-hot 矩阵 (N, K)
        mask = F.one_hot(self.labels, num_classes=self.n_clusters).float()
        # 簇内求和 & 计数
        sums = mask.T @ x                        # (K, D)
        counts = mask.sum(dim=0)                 # (K,)
        # 处理空簇：保留旧中心
        nonzero = counts > 0
        new_centers = torch.zeros_like(self.centers)
        new_centers[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(1)
        new_centers[~nonzero] = self.centers[~nonzero]
        self.centers = new_centers

    def get_assignments(self):
        """Return cluster assignment for each sample."""
        return torch.argmin(self.dists, dim=1)
