import warnings
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None



def get_obs_mem_queue(
    observation_space: spaces.Space,
    buffer_size: int,
    n_envs: int,
    memory_usage: int = 0) -> Dict:
    queue = []
    shapes = {}
    for key, space in observation_space.items():
        queue.append((shapes, key, space))

    while queue:
        parent, key, space = queue.pop(0)
        if isinstance(space, spaces.Box):
            parent[key] = np.zeros((buffer_size, n_envs, *space.shape), dtype=space.dtype)
        elif isinstance(space, spaces.Discrete):
            parent[key] = np.zeros((buffer_size, n_envs, 1), dtype=space.dtype)
        elif isinstance(space, spaces.MultiDiscrete):
            parent[key] =  np.zeros((buffer_size, n_envs, int(len(space.nvec))), dtype=space.dtype) 
        elif isinstance(space, spaces.MultiBinary):
            parent[key] = np.zeros((buffer_size, n_envs, *space.shape), dtype=space.dtype)
        elif isinstance(space, spaces.Dict):
            parent[key] = {}
            for subkey, subspace in space.items():
                queue.append((parent[key], subkey, subspace))        
        else:
            raise NotImplementedError(f"{space} observation space is not supported")

        if not isinstance(space, spaces.Dict):
            memory_usage += parent[key].nbytes
    return shapes

def add_obs_mem_queue(
    memory: Dict,
    pos: int,
    observation: Dict,
    observation_space: spaces.Space,
    n_envs: int,
    copy: bool = False) -> None:
    queue = []
    for key, space in observation_space.items():
        queue.append((memory, observation, key, space))

    while queue:
        parent_mem, parent_obs, key, space = queue.pop(0)

        if isinstance(space, spaces.Discrete):
            parent_obs[key] = parent_obs[key].reshape((n_envs,) + (1,))

        if isinstance(space, spaces.Dict):
            for subkey, subspace in space.items():
                queue.append((parent_mem[key], parent_obs[key], subkey, subspace))
        else:
            if copy:
                parent_mem[key][pos] = parent_obs[key].copy() # np.array()
            else:
                parent_mem[key][pos] = parent_obs[key] # np.array()

def get_flat_mem_queue(
    memory: Dict,
    observation_space: spaces.Space) -> Dict:
    queue = []
    flatten_mem = {}
    for key, space in observation_space.items():
        queue.append((memory, flatten_mem, key, space))

    while queue:
        parent_mem, parent_flat, key, space = queue.pop(0)

        if isinstance(space, spaces.Dict):
            parent_flat[key] = {}
            for subkey, subspace in space.items():
                queue.append((parent_mem[key], parent_flat[key], subkey, subspace))
        else:
            parent_flat[key] = parent_mem[key].reshape((-1,) + space.shape)
    return flatten_mem

def sample_obs_mem_queue(
    memory: Dict,
    batch_inds: np.ndarray,
    env_indices: np.ndarray,
    observation_space: spaces.Space,
    norm_obs_func: Callable,
    env: Optional[VecNormalize],
    to_torch_func: Callable) -> Dict:
    queue = []
    sampled_obs = {}
    for key, space in observation_space.items():
        queue.append((sampled_obs, memory, key, space))

    while queue:
        parent_obs, parent_mem, key, space = queue.pop(0)

        if isinstance(space, spaces.Dict):
            parent_obs[key] = {}
            for subkey, subspace in space.items():
                queue.append((parent_obs[key], parent_mem[key], subkey, subspace))
        else:
            if env_indices is None:
                parent_obs[key] = to_torch_func(norm_obs_func(parent_mem[key][batch_inds], env))
            else:
                parent_obs[key] = to_torch_func(norm_obs_func(parent_mem[key][batch_inds, env_indices, :], env))
    return sampled_obs

class RecDictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer for nested observation spaces
    used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        obs_nbytes = 0
        next_obs_nbytes = 0

        self.observations = get_obs_mem_queue(observation_space, self.buffer_size, self.n_envs, obs_nbytes)
        self.next_observations = get_obs_mem_queue(observation_space, self.buffer_size, self.n_envs, next_obs_nbytes)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            print("obs memory:", obs_nbytes)
            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            print("next obs memory:", next_obs_nbytes)
            total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:  # pytype: disable=signature-mismatch
        # Copy to avoid modification by reference

        add_obs_mem_queue(self.observations,
                          self.pos,
                          obs,
                          self.observation_space,
                          self.n_envs,
                          copy=False)


        add_obs_mem_queue(self.next_observations,
                          self.pos,
                          next_obs,
                          self.observation_space,
                          self.n_envs,
                          copy=True)

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(
        self,
        batch_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_indices),))
        # Convert to torch tensor
        observations = sample_obs_mem_queue(self.observations,
                                            batch_indices,
                                            env_indices,
                                            self.observation_space,
                                            self._normalize_obs,
                                            env,
                                            self.to_torch
                                            )
        next_observations =  sample_obs_mem_queue(self.next_observations,
                                            batch_indices,
                                            env_indices,
                                            self.observation_space,
                                            self._normalize_obs,
                                            env,
                                            self.to_torch
                                            )
        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_indices, env_indices].reshape(-1, 1), env)),
        )
