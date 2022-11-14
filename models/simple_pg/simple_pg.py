# https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

import gym
import torch
import torch.nn as nn
import numpy as np

# from gym.spaces import Discrete, Box
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.optim import Adam


class PolicyEstimator(nn.Module):
  def __init__(self, sizes):
    super().__init__()
    layers = []

    num_layers = len(sizes) - 1
    for i in range(num_layers):
      layers += [nn.Linear(sizes[i], sizes[i+1]), nn.Tanh()]
    layers[-1] = nn.Identity()

    self.model = nn.Sequential(
      *layers
    )

  def forward(self, X):
    out = self.model(X)
    return out


def get_policy(model, obs):
  # get log action probs from observation
  # categorical assumes unnormalized for logits, 
  # so no softmax on model output layer
  logits = model(obs)
  # create distribution from probs to sample from
  policy = Categorical(logits=logits)
  return policy

def get_action(model, obs):
  # get policy for observation
  action_dist = get_policy(model, obs)
  # sample an action from this policy
  action = action_dist.sample().item()
  return action

def get_loss(obs, act, weights, model):
  log_prob = get_policy(model, obs).log_prob(act) 

  # E[log_prob(a|s) * R(tau)]
  return -(log_prob * weights).mean()

def reward_to_go(rewards):
  n = len(rewards)
  rtgs = np.zeros_like(rewards)
  for i in reversed(range(n)):
    rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
  return rtgs

def train_one_epoch(model, env, optimizer, batch_size):
  batch_obs = []
  batch_acts = []
  batch_weights = []
  batch_rets = []
  batch_lens = []

  obs = env.reset()[0]
  done = False
  ep_rewards = []

  while True:
    batch_obs.append(obs.copy())

    act = get_action(model, torch.as_tensor(obs, dtype=torch.float32))
    obs, reward, done, _, _ = env.step(act)

    batch_acts.append(act)
    ep_rewards.append(reward)

    if done:
      ep_ret, ep_len = sum(ep_rewards), len(ep_rewards)
      batch_rets.append(ep_ret)
      batch_lens.append(ep_len)

      # R(tau)
      # batch_weights += [ep_ret] * ep_len
      batch_weights += list(reward_to_go(ep_rewards))

      obs, done, ep_rewards = env.reset()[0], False, []

      if len(batch_obs) > batch_size:
        break

  optimizer.zero_grad()
  batch_loss = get_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                        act=torch.as_tensor(batch_acts, dtype=torch.int32),
                        weights=torch.as_tensor(batch_weights, dtype=torch.float32),
                        model=model)
  batch_loss.backward()
  optimizer.step()
  return batch_loss, batch_rets, batch_lens

def train(model, env, optimizer, batch_size, epochs=50):
  model.train()
  for i in range(epochs):
    batch_loss, batch_rets, batch_lens = train_one_epoch(model, env, optimizer, batch_size)
    print(f'epoch: {i} \t loss: {batch_loss} \t return: {np.mean(batch_rets)} \t ep_len: {np.mean(batch_lens)}')

  return batch_loss, batch_rets
