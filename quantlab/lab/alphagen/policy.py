from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class RecurrentExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, d_model: int = 64, n_layers: int = 1,
                 dropout: float = 0.0, cell: str = "lstm"):
        super().__init__(observation_space, d_model)
        assert isinstance(observation_space, gym.spaces.Box)
        vocab = int(observation_space.high[0]) + 1
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=0)
        cls = nn.GRU if cell.lower() == "gru" else nn.LSTM
        self.rnn = cls(d_model, d_model, n_layers, batch_first=True,
                       dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.long()
        lengths = (obs != 0).sum(1).clamp_min(1)
        output, _ = self.rnn(self.embedding(obs))
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
        return output.gather(1, idx).squeeze(1)
