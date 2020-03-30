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

	def load_experiences(self, experiences: list):
		for exp in experiences:
			self.actions.append(exp['action']),
			self.states.append(exp['state']),
			self.logprobs.append(exp['action_logprob']),
			self.rewards.append(exp['reward']),
			self.is_terminals.append(exp['done']),


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
		state = torch.tensor(state).float().to(device)
		return (res.cpu().data.numpy() for res in self.policy_old.act(state))
	
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
		old_states = torch.squeeze(torch.tensor(np.stack(memory.states)[:, np.newaxis, :],).float().to(device), 1).detach()
		old_actions = torch.squeeze(torch.tensor(np.stack(memory.actions)[:, np.newaxis, :]).float().to(device), 1).detach()
		old_logprobs = torch.squeeze(torch.tensor(np.stack(memory.logprobs)[:, np.newaxis, np.newaxis]).float(), 1).to(device).detach()
		
		# Optimize policy for K epochs:
		tmp_loss = 0
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
	def __init__(self, worker_id, hparams, processing_queue):
		self.id = worker_id
		self.hparams = hparams

		self.processing_queue = processing_queue
		self.inbox_queue = mp.Queue()
		self.experience_queue = mp.Queue()

		self.global_obs = dict()
		self.state = RingBuffer(capacity=self.hparams['state_dim'], dtype=np.float32)

		self.process = mp.Process(
			target=self.run
		)

	def reset(self):
		# flush buffer with all zeros
		self.state.extend(np.zeros((self.hparams['state_dim'])))

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

		if not observations:
			# print("no observations.. there is no one around me?")
			observations = {
				'radarScan': 	[],
			}
			observations.update(old_obs)
		

		# grab the 15 closest tanks
		tank_locations = [ (float(tank_dic['x']), float(tank_dic['y']) ) for tank_dic in observations['radarScan']]
		tank_locations = sorted(tank_locations, key=parse_distances)

		ret_obs = {
			'killCount':	observations['killCount'],
			'hitCount':		observations['hitCount'], 
			'deathCount':	observations['deathCount']
		}

		# List of the closest tank coordinates
		ret_state = np.zeros((self.hparams['num_tanks'],2))

		for i, loc in enumerate(tank_locations):
			ret_state[i] = loc

		return ret_state, ret_obs, False #True if observations['deathCount'] >= 3 else False

	def run(self):
		try:
			while True:
				# clean up
				self.reset()

				obs = self.global_obs

				# RlWorldClient("129.127.147.237", 1337)
				env = RlWorldClient("10.90.159.11", 1337)
				for t in range(self.hparams['max_timesteps']):
					start_time = time.time()

					state_current, obs, done = self.parse_observations(env.read_observation_dict(), obs)
					self.state.extend(state_current.reshape(-1))

					# send state to processing queue
					self.processing_queue.put({
						'id': self.id, 
						'state': np.array(self.state),
					})

					# get action response
					response = self.inbox_queue.get()
					# print(response)
					if response is None:
						print(f"Process {self.id} is shutting down..")
						return

					assert response['id'] == self.id
					action = response['action']
					action_logprob = response['action_logprob']
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

					# Send back experience asap
					action = action.tolist()
					xp = {
						'id': self.id,
						'state': np.array(self.state),
						'action': action,
						'action_logprob': action_logprob,
						'reward': sum(reward),
						'done': done
					}
					# print(xp)
					self.experience_queue.put(xp)
							
					self.global_obs['killCount'] += delta_kills
					self.global_obs['hitCount'] += delta_hits
					self.global_obs['deathCount'] += delta_death
					
					
					
					# Send off action to environment
					env.send_action_dict({
						"name": f"swarm_id:{self.id}",
						"colour": "#7017a1",
						"moveForwardBack": 	action[0],
						"moveRightLeft": 	action[1],
						"turnRightLeft": 	action[2],
						"fire": True,
					})

					if done:
						death_count += 1
						break
					
					# load balancing for the server 
					time_diff = time.time()-start_time
					time.sleep(0 if time_diff > self.hparams['tick_time'] else self.hparams['tick_time']-time_diff)

		except Exception as e:
			print(f"worker {self.id} caught exception: {e}")
			time.sleep(0.5)	

def main():
	mp.set_start_method('spawn')

	############## Hyperparameters ##############
	id = random.randint(0,1000)

	# device = torch.device("cpu")
	print("device:", device)
	env_name = f"{math.floor(time.time())}__id_{id}"
	writer = SummaryWriter("./logs/"+env_name)
	tick_time = 0.20
	max_episodes = 10000        # max training episodes
	max_timesteps = int(15 * (1/tick_time))        # max actions in one episode
	
	
	update_timestep = max_timesteps * 2 # update policy every n timesteps
	action_std = 0.5            # constant std for action distribution (Multivariate Normal)
	K_epochs = 3               # update policy for K epochs
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


	num_producers = 20
	consumer_batch_size = 32

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
	ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

	hparams = {
		'max_timesteps': max_timesteps,
		'state_dim': state_dim,
		'num_tanks': num_tanks,
		'tick_time': tick_time
	}

	producers = {}
	processing_queue = mp.Queue() 

	# create producers as worker objects
	for rank in range(num_producers):
		producers[rank] = Producer(
			rank, 
			hparams,
			processing_queue
			)

	# start prducers
	for worker_id, worker in producers.items():
		worker.process.start()
	print("finished making producer(s)")
		
	# training loop
	for i_episode in tqdm(range(1, max_episodes+1)):
		experiences = []
		jobs_done = 0

		while jobs_done <= update_timestep*num_producers:
			jobs = []

			while (not processing_queue.empty() or len(jobs) == 0) and len(jobs) < consumer_batch_size:
				jobs.append(processing_queue.get())

			if jobs:
				with torch.no_grad():
					data = np.stack([dic['state'].reshape(1, -1) for dic in jobs])
					# print(data.shape)
					action, action_logprob = ppo.select_action(data)
					# print('action', action.shape, action)
					# print('action_logprob', action_logprob.shape, action_logprob)
					for rank, p_action, p_action_logprob in zip((j['id'] for j in jobs), np.rollaxis(action, 0), np.rollaxis(action_logprob, 0)):
						producers[rank].inbox_queue.put({
							'id': rank,
							'action': p_action[0],
							'action_logprob': p_action_logprob[0],
						})
						experiences.append(producers[rank].experience_queue.get())

					jobs_done += len(jobs)
		
		
		# update PPO model once we get enough experiences
		memory.clear_memory()
		memory.load_experiences(experiences)
		loss = ppo.update(memory)

		stats = {
			'rewards/avg_reward': np.stack(memory.rewards).mean(),
			'debug/loss': loss.item(),
		}
		
		# log episode average rewards, stats etc, debug dict
		print(f'Episode {i_episode} \t Avg reward: {stats["rewards/avg_reward"]:.3f}')

		for stat_name, stat in stats.items():
			writer.add_scalar(stat_name, stat, i_episode)

		# torch.save(ppo.policy.state_dict(), "./weights/" + env_name + ".pt")

	print(f"Closing {len(producers)} producers")
	for worker_id, worker in producers.items():
		worker.inbox_queue.put(None)

	for worker_id, worker in producers.items():
		worker.process.join()

	print("All done")

			
if __name__ == '__main__':
	main()