import torch
import torch.nn as nn
import numpy as np


class DangerModel(nn.Module):
    def __init__(self, label_vocab):
        super().__init__()

        self.label_vocab = label_vocab
        self.label_to_index = {
            label: idx for idx, label in enumerate(label_vocab)
        }

        input_dim = len(label_vocab) + 2  # one-hot + confidence + bbox_area_ratio

        # Lightweight MLP
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()   # Output ∈ [0, 1]
        )

        # Inference mode
        self.eval()

    # ---------------- Feature encoding ----------------

    def _one_hot_encode(self, label):
        vec = np.zeros(len(self.label_vocab), dtype=np.float32)
        if label in self.label_to_index:
            vec[self.label_to_index[label]] = 1.0
        return vec

    def _build_feature_vector(self, label, confidence, bbox_area_ratio):
        label_vec = self._one_hot_encode(label)
        features = np.concatenate(
            [label_vec, [confidence, bbox_area_ratio]]
        )
        return torch.tensor(features, dtype=torch.float32)

    # ---------------- Public API ----------------

    def predict(self, label, confidence, bbox_area_ratio):
        """
        Returns danger_score ∈ [0, 1]
        """
        with torch.no_grad():
            x = self._build_feature_vector(
                label, confidence, bbox_area_ratio
            )
            score = self.model(x)
            return score.item()
