from __future__ import annotations

import gymnasium as gym
import numpy as np
from copy import deepcopy
from stable_baselines3.common.vec_env import DummyVecEnv

from .expressions import CONSTANTS, OPERATORS, WINDOWS, FeatureType, RPNBuilder, Token
from .pool import AlphaPool


TOKENS = (
    tuple(Token("operator", x) for x in OPERATORS)
    + tuple(Token("feature", x.name) for x in FeatureType)
    + tuple(Token("constant", x) for x in CONSTANTS)
    + tuple(Token("window", x) for x in WINDOWS)
    + (Token("stop", "STOP"),)
)


class AlphaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, pool: AlphaPool, max_length: int = 15, defer_evaluation: bool = False):
        super().__init__()
        self.pool, self.max_length = pool, max_length
        self.defer_evaluation = defer_evaluation
        self.action_space = gym.spaces.Discrete(len(TOKENS))
        self.observation_space = gym.spaces.Box(0, len(TOKENS), (max_length,), dtype=np.int16)
        self.builder = RPNBuilder()
        self.state = np.zeros(max_length, dtype=np.int16)
        self.position = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.builder = RPNBuilder()
        self.state = np.zeros(self.max_length, dtype=np.int16)
        self.position = 0
        return self.state.copy(), {}

    def action_masks(self) -> np.ndarray:
        mask = np.array([self.builder.valid(token) for token in TOKENS], dtype=bool)
        if self.position < 2: mask[-1] = False
        return mask

    def step(self, action: int):
        token = TOKENS[int(action)]
        if not self.builder.valid(token):
            return self.state.copy(), -1.0, True, False, {"invalid_action": str(token)}
        terminated = token.kind == "stop"
        reward = 0.0
        info = {}
        if terminated:
            expr = self.builder.expression()
            if self.defer_evaluation: info["candidate"] = (expr, self.builder.token_string)
            else: reward = self.pool.try_candidate(expr, self.builder.token_string)
        else:
            self.builder.add(token)
            self.state[self.position] = int(action) + 1  # 0 专门留给 padding
            self.position += 1
            if self.position >= self.max_length:
                if self.builder.valid(TOKENS[-1]):
                    expr = self.builder.expression()
                    if self.defer_evaluation: info["candidate"] = (expr, self.builder.token_string)
                    else: reward = self.pool.try_candidate(expr, self.builder.token_string)
                else:
                    reward = -1.0
                terminated = True
        return self.state.copy(), float(reward), terminated, False, info


class BatchedAlphaVecEnv(DummyVecEnv):
    """批量策略推理，并把同一 vector step 完成的表达式一起并行求值。"""
    def __init__(self, pool: AlphaPool, n_envs: int, max_length: int):
        self.pool = pool
        super().__init__([lambda: AlphaEnv(pool, max_length, defer_evaluation=True) for _ in range(n_envs)])

    def step_wait(self):
        pending: list[tuple[int, tuple]] = []
        final_obs = {}
        for env_idx in range(self.num_envs):
            obs, reward, terminated, truncated, info = self.envs[env_idx].step(self.actions[env_idx])
            self.buf_rews[env_idx] = reward
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx] = info
            if "candidate" in info: pending.append((env_idx, info.pop("candidate")))
            if self.buf_dones[env_idx]: final_obs[env_idx] = obs
            else: self._save_obs(env_idx, obs)
        if pending:
            rewards = self.pool.try_candidates_batch([candidate for _, candidate in pending])
            for (env_idx, _), reward in zip(pending, rewards): self.buf_rews[env_idx] = reward
        for env_idx in range(self.num_envs):
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["TimeLimit.truncated"] = False
                self.buf_infos[env_idx]["terminal_observation"] = final_obs[env_idx]
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                self._save_obs(env_idx, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)
