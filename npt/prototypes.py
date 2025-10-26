import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPrototypes(nn.Module):
    """Learnable prototype set with a lightweight Information-Bottleneck-style
    surrogate objective.

    This module is intentionally small and self-contained so it can be used in
    benchmark experiments. It exposes:
      - prototypes: nn.Parameter of shape (n_prototypes, prototype_dim)
      - forward(x): returns soft assignments p(z|x) and logits
      - ib_loss(x, labels, relevance_weight, redundancy_weight): returns
        a surrogate IB-style loss combining relevance (H(Y|Z)) and a
        redundancy penalty among prototypes.

    Notes / limitations:
    - The mutual-information objective is approximated using cluster-wise
      empirical estimates over the provided batch (surrogate). For production
      experiments you'd want to accumulate running statistics over the
      training set or use a more principled MI estimator.
    - Prototypes are generic vectors; if you want per-feature prototypes
      (matching the NPT column encodings) you'll need to adapt the shapes.
    """

    def __init__(self, n_prototypes: int, prototype_dim: int,
                 device: str = 'cpu'):
        super().__init__()
        self.n_prototypes = int(n_prototypes)
        self.prototype_dim = int(prototype_dim)
        self.device = device
        # Initialize prototypes randomly; users can call init_from_data()
        self.prototypes = nn.Parameter(
            torch.randn(self.n_prototypes, self.prototype_dim, device=self.device)
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        """Compute soft assignments p(z|x) and similarity logits.

        Args:
            x: Tensor of shape (B, prototype_dim)
            temperature: temperature scaling applied to logits (float > 0)
        Returns:
            p_z_given_x: (B, P) soft assignments
            logits: (B, P) similarity logits (negative squared distance / temperature)
        """
        if x.dim() != 2 or x.size(1) != self.prototype_dim:
            x = x.view(x.size(0), -1)
            assert x.size(1) == self.prototype_dim, (
                f"Input dim {x.size(1)} != prototype_dim {self.prototype_dim}")

        dists = torch.cdist(x, self.prototypes, p=2)
        d2 = dists ** 2
        logits = -d2 / float(temperature)
        p = F.softmax(logits, dim=1)
        return p, logits

    def ib_loss(self, x: torch.Tensor, labels: torch.Tensor,
                relevance_weight: float = 1.0,
                redundancy_weight: float = 1.0,
                compression_weight: float = 0.1,
                temperature: float = 1.0,
                eps: float = 1e-8):
        """Compute a surrogate IB-style loss on the batch.

        This version reuses `forward` to compute temperature-scaled soft
        assignments, clamps p(z) for stability, uses Laplace smoothing when
        estimating p(y|z), and applies a squared cosine-similarity penalty for
        redundancy (stronger penalty than absolute similarity).
        """

        # Soft assignments with temperature (reuse forward)
        p_z_x, _ = self.forward(x, temperature=temperature)
        B, P = p_z_x.shape

        # Convert labels to integer class indices if necessary
        if labels.dim() == 2:
            labels_int = torch.argmax(labels, dim=1)
        else:
            labels_int = labels.long()
        C = int(labels_int.max().item()) + 1 if labels_int.numel() > 0 else 1

        # One-hot labels (B, C)
        y_onehot = F.one_hot(labels_int, num_classes=C).float()

        # Empirical p(z) with stability clamp
        p_z = p_z_x.mean(dim=0).clamp_min(eps)  # (P,)

        # p(y|z) with Laplace smoothing
        alpha = 0.01
        numerator = p_z_x.t() @ y_onehot + alpha  # (P, C)
        denominator = p_z_x.sum(dim=0, keepdim=True).t() + alpha * C  # (P, 1)
        p_y_given_z = numerator / denominator.clamp_min(eps)

        # Relevance: H(Y|Z) - minimize to maximize I(Z;Y)
        H_y_given_z = -(p_y_given_z * torch.log(p_y_given_z + eps)).sum(dim=1)  # (P,)
        relevance = (p_z * H_y_given_z).sum()

        # Compression: H(Z) - maximize to prevent collapse
        H_z = -(p_z * torch.log(p_z + eps)).sum()
        compression = -H_z

        # Redundancy: Penalize prototype similarity (cosine)
        if P > 1:
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)
            sim_matrix = proto_norm @ proto_norm.t()
            mask = 1.0 - torch.eye(P, device=self.prototypes.device)
            redundancy = (sim_matrix * mask).pow(2).mean()  # Square for stronger penalty
        else:
            redundancy = torch.tensor(0.0, device=self.prototypes.device)

        # Combined loss
        loss = (relevance_weight * relevance + 
                redundancy_weight * redundancy + 
                compression_weight * compression)

        info = {
            'relevance': float(relevance.detach().cpu().item()),
            'redundancy': float(redundancy.detach().cpu().item()) if isinstance(redundancy, torch.Tensor) else 0.0,
            'compression': float(compression.detach().cpu().item()),
            'H_z': float(H_z.detach().cpu().item())
        }

        return loss, info

    def init_from_data(self, X: np.ndarray = None, method: str = 'random', random_state: int = 0):
        """Initialize prototypes from data.

        Args:
            X: numpy array (N, D) with D == prototype_dim
            method: 'random' or 'kmeans' (requires sklearn)
        """
        if X is None:
            # keep random init
            return
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == self.prototype_dim
        if method == 'random':
            idx = np.random.RandomState(random_state).choice(X.shape[0], size=self.n_prototypes, replace=True)
            proto = torch.from_numpy(X[idx].astype(np.float32)).to(self.prototypes.device)
            with torch.no_grad():
                self.prototypes.copy_(proto)
        elif method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
            except Exception:
                raise RuntimeError('sklearn required for kmeans initialization')
            kmeans = KMeans(n_clusters=self.n_prototypes, random_state=random_state)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            proto = torch.from_numpy(centers.astype(np.float32)).to(self.prototypes.device)
            with torch.no_grad():
                self.prototypes.copy_(proto)
        else:
            raise ValueError('Unknown init method: ' + str(method))

    def state_dict_for_saving(self):
        return {'prototypes': self.prototypes.detach().cpu().numpy()}

    def load_state_dict_from_saved(self, d: dict):
        arr = np.asarray(d.get('prototypes'))
        assert arr.shape == (self.n_prototypes, self.prototype_dim)
        with torch.no_grad():
            self.prototypes.copy_(torch.from_numpy(arr).to(self.prototypes.device))
