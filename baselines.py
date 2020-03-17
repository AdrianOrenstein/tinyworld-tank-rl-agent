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

from torch.utils.tensorboard import SummaryWriter

id = random.randint(0,1000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device:", device)


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

	def save_memory(self):
		time_id = math.floor(time.time())
		# store what we're deleting
		with open(f"./data/{time_id}__{id}.pkl", 'wb') as f:
			pickle.dump({
				"actions": self.actions[:],
				"states": self.states[:],
				"logprobs": self.logprobs[:],
				"rewards": self.rewards[:],
				"is_terminals": self.is_terminals[:],
				}, f)

	def find_experiences(self, seconds_ago=15):
		now = dt.datetime.now()
		ago = now-dt.timedelta(seconds=seconds_ago)

		found_experiences = 0

		try:
			for root, dirs, files in os.walk('./data/'):  
				for fname in files:
					path = os.path.join(root, fname)
					st = os.stat(path)    
					mtime = dt.datetime.fromtimestamp(st.st_mtime)

					filename = path.split('/')[-1]
					filename_ext = filename[-3:]
					filename_id = filename.split('__')[-1][:3]

					if mtime > ago:
						# if it's not our experience
						if filename_id != str(id):
							self.load_memory(path)
							found_experiences += 1

					else:
						# cleaning up our own mess
						if mtime > now-dt.timedelta(seconds=seconds_ago*8) and filename_id == str(id):
							print("deleting:", filename, filename_ext, filename_id)
							os.remove(path)
		except Exception as e:
			print("caught exception:", e)

		return found_experiences

	def load_memory(self, path):
		print("loading experience:", path)
		new_experiences = pickle.load(open(path, "rb"))

		self.actions.extend(new_experiences['actions'])
		self.states.extend(new_experiences['states'])
		self.logprobs.extend(new_experiences['logprobs'])
		self.rewards.extend(new_experiences['rewards'])
		self.is_terminals.extend(new_experiences['is_terminals'])



class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, action_std):
		super(ActorCritic, self).__init__()
		# action mean range -1 to 1
		self.actor =  nn.Sequential(
				nn.Linear(state_dim, 128),
				nn.LeakyReLU(),
				nn.Linear(128, 128),
				nn.LeakyReLU(),
				nn.Linear(128, action_dim),
				nn.LeakyReLU(),
				)
		# critic
		self.critic = nn.Sequential(
				nn.Linear(state_dim, 128),
				nn.LeakyReLU(),
				nn.Linear(128, 128),
				nn.LeakyReLU(),
				nn.Linear(128, 1)
				)
		self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
		
	def forward(self):
		raise NotImplementedError
	
	def act(self, state, memory):
		action_mean = self.actor(state)
		cov_mat = torch.diag(self.action_var).to(device)
		
		dist = MultivariateNormal(action_mean, cov_mat)
		action = dist.sample()
		action_logprob = dist.log_prob(action)
		
		memory.states.append(state)
		memory.actions.append(action)
		memory.logprobs.append(action_logprob)
		
		return action.detach()
	
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
		
		self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
		
		self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		self.MseLoss = nn.MSELoss()
	
	def select_action(self, state, memory):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
	
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
		
def main():
	############## Hyperparameters ##############
	env_name = f"{math.floor(time.time())}__id_{id}"
	writer = SummaryWriter("./logs/"+env_name)
	tick_time = 0.1
	log_interval = 10           # print avg reward in the interval
	max_episodes = 10000        # max training episodes
	max_timesteps = int(20 * (1/tick_time))        # max actions in one episode
	
	
	update_timestep = max_timesteps * 2 # update policy every n timesteps
	action_std = 0.50            # constant std for action distribution (Multivariate Normal)
	K_epochs = 80               # update policy for K epochs
	eps_clip = 0.5              # clip parameter for PPO
	gamma = 0.99                # discount factor
	
	lr = 0.0003                 # parameters for Adam optimizerd
	betas = (0.9, 0.999)
	
	random_seed = None
	#############################################
	
	# creating environment
	num_captures = 3
	num_tanks = 15
	action_dim = 2
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
	ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
	print(lr,betas)
	
	# logging variables
	running_reward = 0
	avg_length = 0
	time_step = 0

	def get_dist(x: float, y: float) -> float:
		return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

	def parse_distances(arr: tuple):
		return get_dist(arr[0], arr[1])

	def parse_observations(observations: dict, old_obs: dict) -> (list, int, bool):
		""" 
		parse observation dict into desirable data structures
		"""

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
		ret_state = np.zeros((num_tanks,2))

		for i, loc in enumerate(tank_locations):
			ret_state[i] = loc

		return ret_state, ret_obs, True if observations['deathCount'] >= 3 else False

	# training loop
	for i_episode in tqdm(range(1, max_episodes+1)):
		batch_time_begin = time.time()
		try:
			env = RlWorldClient("129.127.147.237", 1337)
			state = RingBuffer(capacity=state_dim, dtype=np.float32)
			# instantiate buffer with all zeros
			state.extend(np.zeros((state_dim)))

			global_obs = {
				'killCount':	0,
				'hitCount': 	0,
				'deathCount':	0,
			}
			obs = global_obs
			for t in range(max_timesteps):
				time_step += 1
				start_time = time.time()

				state_current, obs, done = parse_observations(env.read_observation_dict(), obs)
				state.extend(state_current.reshape(-1))

				action = ppo.select_action(np.array(state), memory)
				state.extend(action.reshape(-1))

				# calculate rewards
				mm_clip = lambda x, l, u: max(l, min(u, x))

				# Rewards
				delta_kills = obs['killCount']	- global_obs['killCount']
				delta_hits 	= obs['hitCount']	- global_obs['hitCount']
				delta_death = obs['deathCount']	- global_obs['deathCount']

				tmp_distances = (get_dist(x_cord, y_cord) for x_cord, y_cord in zip(state[-num_tanks:-num_tanks+1:2], state[-num_tanks+1:-num_tanks+1:2]))
				reward_distance = sum((mm_clip(1/dist, 0, 1) if dist>0 else 0 for dist in tmp_distances))

				reward = sum([
					5	 * delta_kills,
					1    * delta_hits,
					0.01 * reward_distance, # was 1 instead of 0.1
					# -0.5 * delta_death
				])
				
				global_obs['killCount'] += delta_kills
				global_obs['hitCount'] += delta_hits
				global_obs['deathCount'] += delta_death

				action = [mm_clip(float(act), -1, 1) for act in action]
				# print(
				# 	"action:", action, "\n",
				# 	"state:", state, "\n", 
				# 	"global_obs:", global_obs, "\n", 
				# 	"done:", done, "\n",
				# 	"reward:", reward, "\n",
				# )

				# Apply action to dict
				action_dict = {
					"name": f"swarm_v2_ep:{i_episode}={(i_episode/(max_episodes+1))*100:.1f}%_id:{id}",
					"colour": "#7017a1",
					"moveForwardBack": 	action[0],
					"moveRightLeft": 	0,
					"turnRightLeft": 	action[1],
					"fire": True, # action[3] > 0,
				}

				env.send_action_dict(action_dict)

				# Saving reward and is_terminals:
				memory.rewards.append(reward)
				memory.is_terminals.append(done)

				running_reward += reward

				if done:
					death_count += 1
					break

				avg_length += t
				
				# sleep 
				end_time = time.time()
				time_diff = end_time-start_time
				time.sleep(0 if time_diff > tick_time else tick_time-time_diff)
		
			writer.add_scalar('rewards/ep_killCount', global_obs['killCount'], i_episode)
			writer.add_scalar('rewards/ep_hitCount', global_obs['hitCount'], i_episode)
			writer.add_scalar('rewards/ep_deathCount', global_obs['deathCount'], i_episode)

			if i_episode % log_interval == 0 and len(memory.states) > 0:
				# load memory from other saved batches
				batch_time_end = time.time()
				batch_time_diff = batch_time_end - batch_time_begin
				# print("batch_time_diff", batch_time_diff)
				memory.save_memory()
				found_count = memory.find_experiences(seconds_ago=batch_time_diff*1.95)
				loss = ppo.update(memory)
				memory.clear_memory()
				time_step = 0
				
				# logging
				# if i_episode % log_interval == 0:
				avg_length = avg_length//log_interval
				running_reward = running_reward/log_interval
				
				print(f'Episode {i_episode} \t Avg length: {avg_length} \t Avg reward: {running_reward:.3f}')
				writer.add_scalar('rewards/reward', running_reward, i_episode)
				writer.add_scalar('rewards/avg_length', avg_length, i_episode)
				writer.add_scalar('debug/found_exp', found_count, i_episode)
				writer.add_scalar('debug/loss', -float(loss.sum().detach()), i_episode)
				running_reward = 0
				avg_length = 0
				torch.save(ppo.policy.state_dict(), "./weights/" + env_name + ".pt")

		except Exception as e:
			print("caught exception:", e)
			time.sleep(1)
	
			
if __name__ == '__main__':
	main()

# while True:
#     try:
#         client = RlWorldClient("129.127.147.237",1337)
#         while True:
#             # print("read")
#             print(client.read_observation_dict())
#             # print("send")

#             action_dict = {
#                 "name": "Adrian " + str(id),
#                 "colour": "#7017a1",
#                 "moveForwardBack":1.0,
#                 "moveRightLeft": 0.0,
#                 "turnRightLeft": random.uniform(-1.0,1.0),
#                 "fire": True,
#             }

#             client.send_action_dict(action_dict)
#             # print("sleep")
#             time.sleep(0.5)
#     except Exception as e:
#         print(e)
#         time.sleep(1)