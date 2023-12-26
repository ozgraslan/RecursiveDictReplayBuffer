# RecursiveDictReplayBuffer
Recursive implementation of [Dictionary Replay Buffer](https://github.com/DLR-RM/stable-baselines3/blob/373166d6ac30561c378bdd46e8dba4ef0760f996/stable_baselines3/common/buffers.py#L523) from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). For nested observation spaces.


# Implementation Details

- The main idea comes from stable_baselines3.common.preprocessing.get_obs_shape
which recursively extract observation spaces's shape information. 
Transformed that to a queue-based implementation.
- Works with gym.vector.SyncVectorEnv
  - DummyVecEnv does not support nested observations.
  - Did not test with SubprocVecEnv. 
- Queue-based implementation of 
  - creating memory 
  - adding obs to memory 
  - sampling from memory
  - concatenating two nested observations
  - normalizing observations and converting to torch


