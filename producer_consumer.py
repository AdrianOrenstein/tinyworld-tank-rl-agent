import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
from numpy_ringbuffer import RingBuffer

from rlworldclient import RlWorldClient
import time
import os
import datetime as dt
import random
import math
import pickle

from tqdm.auto import tqdm

from itertools import repeat

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.is_terminals = []
	
	def clear_memory(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.is_terminals[:]


class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, action_std):
		super(ActorCritic, self).__init__()
		# action mean range -1 to 1
		self.hidden_size = 64
		self.actor =  nn.Sequential(
				nn.Linear(state_dim, self.hidden_size),
				nn.ReLU(),
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.ReLU(),
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.ReLU(),
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.ReLU(),
				nn.Linear(self.hidden_size, action_dim),
				nn.Tanh(),
				).to(device)

		# critic
		self.critic = nn.Sequential(
				nn.Linear(state_dim, self.hidden_size),
				nn.ReLU(),
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.ReLU(),
				nn.Linear(self.hidden_size, 1)
				).to(device)

		self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
		
	def forward(self):
		raise NotImplementedError
	
	def act(self, state):
		action_mean = self.actor(state)
		cov_mat = torch.diag(self.action_var).to(device)
		
		dist = MultivariateNormal(action_mean, cov_mat)
		action = dist.sample()
		action_logprob = dist.log_prob(action)
		
		return action.detach(), action_logprob.detach()
	
	def evaluate(self, state, action):   
		action_mean = self.actor(state)
		
		action_var = self.action_var.expand_as(action_mean)
		cov_mat = torch.diag_embed(action_var).to(device)
		
		dist = MultivariateNormal(action_mean, cov_mat)
		
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_value = self.critic(state)
		
		return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
	def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
		self.lr = lr
		self.betas = betas
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs
		
		self.policy = ActorCritic(state_dim, action_dim, action_std)
		self.policy.share_memory()
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
		
		self.policy_old = ActorCritic(state_dim, action_dim, action_std)
		self.policy_old.share_memory()
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		self.MseLoss = nn.MSELoss()
	
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return (res.cpu().data.numpy().flatten() for res in self.policy_old.act(state))
	
	def update(self, memory):
		# Monte Carlo estimate of rewards:
		print("Updating networks...")
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
		
		# Normalizing the rewards:
		rewards = torch.tensor(rewards).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
		
		# convert list to tensor
		old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
		old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
		old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

		min_size = min(len(rewards), len(old_states), len(old_actions), len(old_logprobs))

		rewards = rewards[:min_size]
		old_states = old_states[:min_size]
		old_actions = old_actions[:min_size]
		old_logprobs = old_logprobs[:min_size]
		
		tmp_loss = 0
		# Optimize policy for K epochs:
		for _ in range(self.K_epochs):
			# Evaluating old actions and values :
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
			
			# Finding the ratio (pi_theta / pi_theta__old):
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss:
			advantages = rewards - state_values 
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
			loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
			
			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()
			tmp_loss += loss.mean()
			self.optimizer.step()
			
		# Copy new weights into old policy:
		# print("updating policy, loss:", loss/self.K_epochs)
		self.policy_old.load_state_dict(self.policy.state_dict())

		return tmp_loss / self.K_epochs

import torch.multiprocessing as mp
import ctypes
import numpy as np


# Parallel processing
class Producer:
	def __init__(self, worker_id, worker_ppo, hparams):
		self.id = worker_id
		self.global_obs = dict()
		self.ppo = worker_ppo
		self.memory = Memory()

		self.hparams = hparams
		self.state = RingBuffer(capacity=self.hparams['state_dim'], dtype=np.float32)

	def reset(self):
		# instantiate buffer with all zeros
		self.state.extend(np.zeros((self.hparams['state_dim'])))

		self.memory.clear_memory()

		self.global_obs = {
			'killCount':	0,
			'hitCount': 	0,
			'deathCount':	0,
		}

	def parse_observations(self, observations: dict, old_obs: dict) -> (list, int, bool):
		""" 
		parse observation dict into desirable data structures
		"""

		def get_dist(x: float, y: float) -> float:
			return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

		def parse_distances(arr: tuple):
			return get_dist(arr[0], arr[1])

		# print(observations)

		if not observations:
			# print("no observations.. there is no one around me?")
			observations = {
				'radarScan': 	[],
			}
			observations.update(old_obs)
		

		# grab the 15 closest tanks
		tank_locations = [ (float(tank_dic['x']), float(tank_dic['y']) ) for tank_dic in observations['radarScan']]
		tank_locations = sorted(tank_locations, key=parse_distances)
		# print([(parse_distances(dis), 1/parse_distances(dis)) for dis in tank_locations])

		ret_obs = {
			'killCount':	observations['killCount'],
			'hitCount':		observations['hitCount'], 
			'deathCount':	observations['deathCount']
		}

		# List of the closest tank coordinates
		ret_state = np.zeros((self.hparams['num_tanks'],2))

		for i, loc in enumerate(tank_locations):
			ret_state[i] = loc

		return ret_state, ret_obs, True if observations['deathCount'] >= 3 else False

	def run(self):
		# clean up
		self.reset()

		obs = self.global_obs

		# RlWorldClient("129.127.147.237", 1337)
		env = RlWorldClient("10.90.159.11", 1337)
		for t in range(self.hparams['max_timesteps']):
			# try:
			start_time = time.time()

			state_current, obs, done = self.parse_observations(env.read_observation_dict(), obs)
			self.state.extend(state_current.reshape(-1))

			action, action_logprob = self.ppo.select_action(np.array(self.state))
			self.state.extend(action.reshape(-1))

			# calculate rewards
			mm_clip = lambda x, l, u: max(l, min(u, x))

			# Rewards
			delta_kills = obs['killCount']	- self.global_obs['killCount']
			delta_hits 	= obs['hitCount']	- self.global_obs['hitCount']
			delta_death = obs['deathCount']	- self.global_obs['deathCount']

			time_discount = mm_clip((-1/(self.hparams['max_timesteps']*1.1)) * t + 1, 0, 1)

			reward = [
				5	 	* delta_kills * time_discount,
				1    	* delta_hits * time_discount,
			]
					
			self.global_obs['killCount'] += delta_kills
			self.global_obs['hitCount'] += delta_hits
			self.global_obs['deathCount'] += delta_death

			
			# Apply action to dict
			action = [float(act) for act in action]
			
			action_dict = {
				"name": f"swarm_id:{self.id}",
				"colour": "#7017a1",
				"moveForwardBack": 	action[0],
				"moveRightLeft": 	action[1],
				"turnRightLeft": 	action[2],
				"fire": True,
			}

			env.send_action_dict(action_dict)

			# Saving all memory:
			self.memory.states.append(self.state),
			self.memory.actions.append(action),
			self.memory.logprobs.append(action_logprob),
			self.memory.rewards.append(sum(reward)),
			self.memory.is_terminals.append(done),

			if done:
				death_count += 1
				break
			
			# load balancing for the server 
			end_time = time.time()
			time_diff = end_time-start_time
			time.sleep(0 if time_diff > self.hparams['tick_time'] else self.hparams['tick_time']-time_diff)

			# except Exception as e:
			# 	print(f"worker {self.id} caught exception: {e}")
			# 	time.sleep(0.5)

		return self.memory	

def run_worker(bot_id, ppo, hparams):
	producer = Producer(bot_id, ppo, hparams)
	return producer.run()

def main():
	############## Hyperparameters ##############
	id = random.randint(0,1000)

	# device = torch.device("cpu")
	print("device:", device)
	env_name = f"{math.floor(time.time())}__id_{id}"
	writer = SummaryWriter("./logs/"+env_name)
	tick_time = 0.20
	batch_size = 30 # number of experiences until we run PPO
	max_episodes = 10000        # max training episodes
	max_timesteps = int(15 * (1/tick_time))        # max actions in one episode
	
	
	update_timestep = max_timesteps * 2 # update policy every n timesteps
	action_std = 0.5            # constant std for action distribution (Multivariate Normal)
	K_epochs = 10               # update policy for K epochs
	eps_clip = 0.2              # clip parameter for PPO
	gamma = 0.99                # discount factor
	
	lr = 0.0003                 # parameters for Adam optimizerd
	betas = (0.9, 0.999)
	
	random_seed = None
	
	# creating environment
	num_captures = 3
	num_tanks = 15
	action_dim = 3
	state_dim = (num_tanks*2)*(num_captures)+(action_dim*num_captures) # 15 tanks with (x,y) * 3 captures
	

	print(
		"state_dim:", state_dim, "\n",
		"action_dim:", action_dim, "\n",
		"max_timesteps:", max_timesteps, "\n",
		"update_timestep:", update_timestep, "\n",
	)
	
	if random_seed:
		print("Random Seed: {}".format(random_seed))
		torch.manual_seed(random_seed)
		env.seed(random_seed)
		np.random.seed(random_seed)
	
	memory = Memory()
	mp.set_start_method('forkserver')
	ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
		
	# training loop
	for i_episode in tqdm(range(1, max_episodes+1)):

		# create workers, give them access to ppo inference
		num_workers = 8

		hparams = {
			'max_timesteps': max_timesteps,
			'state_dim': state_dim,
			'num_tanks': num_tanks,
			'tick_time': tick_time
		}

		# bots = [Producer(bot_id, ppo, hparams) for bot_id in range(num_workers)]
		processes = []
		for rank in range(num_workers):
			p = mp.Process(target=run_worker, args=(rank, ppo, hparams))
			p.start()
			processes.append(p)

		for p in processes:
			p.join()
		# return memory

		print(p)
		break

		# update PPO model once we reach Y batch size 


		# log episode average rewards, stats etc, debug dict
	
			
if __name__ == '__main__':
	main()