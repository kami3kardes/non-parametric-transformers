import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPrototypes(nn.Module):
    """Learnable prototype set with an Information-Bottleneck-style surrogate objective.

    Features:
        - prototypes: nn.Parameter of shape (n_prototypes, prototype_dim)
        - forward(x): returns soft assignments p(z|x) and similarity logits
        - ib_loss(x, labels, ...): computes IB-style loss
        - init_from_data: initialize prototypes from data
        - get_prototype_indices: returns indices for sampler compatibility
        - state_dict/load_state_dict: PyTorch-native saving and legacy helpers
    """

    def __init__(self, n_prototypes: int, prototype_dim: int, device: str = 'cpu'):
        super().__init__()
        self.n_prototypes = int(n_prototypes)
        self.prototype_dim = int(prototype_dim)
        self.device = torch.device(device)
        self.prototypes = nn.Parameter(
            torch.randn(self.n_prototypes, self.prototype_dim, device=self.device)
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        x = x.to(self.prototypes.device)
        if x.dim() != 2 or x.size(1) != self.prototype_dim:
            x = x.view(x.size(0), -1)
            assert x.size(1) == self.prototype_dim, (
                f"Input dim {x.size(1)} != prototype_dim {self.prototype_dim}"
            )

        dists = torch.cdist(x, self.prototypes, p=2)
        logits = -dists ** 2 / float(temperature)
        p_z_x = F.softmax(logits, dim=1)
        return p_z_x, logits

    def ib_loss(self, x: torch.Tensor, labels: torch.Tensor,
                relevance_weight: float = 1.0,
                redundancy_weight: float = 1.0,
                compression_weight: float = 0.1,
                temperature: float = 1.0,
                eps: float = 1e-8):
        x = x.to(self.prototypes.device)
        labels = labels.to(self.prototypes.device)

        p_z_x, _ = self.forward(x, temperature)
        B, P = p_z_x.shape

        if labels.dim() == 2:
            labels_int = torch.argmax(labels, dim=1)
        else:
            labels_int = labels.long()

        C = int(labels_int.max().item()) + 1 if labels_int.numel() > 0 else 1
        y_onehot = F.one_hot(labels_int, num_classes=C).float()

        p_z = p_z_x.mean(dim=0).clamp_min(eps)
        alpha = 0.01
        numerator = p_z_x.t() @ y_onehot + alpha
        denominator = p_z_x.sum(dim=0, keepdim=True).t() + alpha * C
        p_y_given_z = numerator / denominator.clamp_min(eps)

        # Relevance: H(Y|Z)
        H_y_given_z = -(p_y_given_z * torch.log(p_y_given_z + eps)).sum(dim=1)
        relevance = (p_z * H_y_given_z).sum()

        # Compression: H(Z)
        H_z = -(p_z * torch.log(p_z + eps)).sum()
        compression = -H_z

        # Redundancy: squared cosine similarity between prototypes
        if P > 1:
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)
            sim_matrix = proto_norm @ proto_norm.t()
            mask = 1.0 - torch.eye(P, device=self.prototypes.device)
            redundancy = (sim_matrix * mask).pow(2).mean()
        else:
            redundancy = torch.tensor(0.0, device=self.prototypes.device)

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
        if X is None:
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
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            proto = torch.from_numpy(centers.astype(np.float32)).to(self.prototypes.device)
            with torch.no_grad():
                self.prototypes.copy_(proto)
        else:
            raise ValueError(f"Unknown init method: {method}")

    def get_prototype_indices(self):
        """Return prototype indices for sampler integration."""
        return np.arange(self.n_prototypes, dtype=np.int32)

    # PyTorch-native state_dict integration (non-destructive)
    def state_dict(self, *args, **kwargs):
        # Return a simple dict that can be saved by torch.save()
        base = super().state_dict(*args, **kwargs)
        # include prototypes as float32 cpu tensor for portability
        base['prototypes'] = self.prototypes.detach().cpu()
        return base

    # Legacy helpers used elsewhere in repo
    def state_dict_for_saving(self):
        return {'prototypes': self.prototypes.detach().cpu().numpy()}

    def load_state_dict_from_saved(self, d: dict):
        arr = np.asarray(d.get('prototypes'))
        arr_t = torch.as_tensor(arr, device=self.prototypes.device)
        assert arr_t.shape == (self.n_prototypes, self.prototype_dim)
        with torch.no_grad():
            self.prototypes.copy_(arr_t)
