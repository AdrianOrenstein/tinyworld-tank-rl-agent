import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

from rlworldclient import RlWorldClient
import time
import random
import math

id = random.randint(0,100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("CUDA?", torch.cuda.is_available())
# device = torch.device("cpu")

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
		self.actor =  nn.Sequential(
				nn.Linear(state_dim, 64),
				nn.Tanh(),
				nn.Linear(64, 32),
				nn.Tanh(),
				nn.Linear(32, action_dim),
				nn.Tanh()
				)
		# critic
		self.critic = nn.Sequential(
				nn.Linear(state_dim, 64),
				nn.Tanh(),
				nn.Linear(64, 32),
				nn.Tanh(),
				nn.Linear(32, 1)
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
		
		# Optimize policy for K epochs:
		for _ in range(self.K_epochs):
			# Evaluating old actions and values :
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
			
			# Finding the ratio (pi_theta / pi_theta__old):
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss:
			advantages = rewards - state_values.detach()   
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
			loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
			
			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()
			
		# Copy new weights into old policy:
		self.policy_old.load_state_dict(self.policy.state_dict())
		
def main():
	############## Hyperparameters ##############
	tick_time = 0.1
	solved_reward = 300         # stop training if avg_reward > solved_reward
	log_interval = 20           # print avg reward in the interval
	max_episodes = 10000        # max training episodes
	max_timesteps = int(50 * (1/tick_time))        # max actions in one episode
	
	
	update_timestep = max_timesteps * 1      # update policy every n timesteps
	action_std = 0.5            # constant std for action distribution (Multivariate Normal)
	K_epochs = 80               # update policy for K epochs
	eps_clip = 0.2              # clip parameter for PPO
	gamma = 0.99                # discount factor
	
	lr = 0.0003                 # parameters for Adam optimizer
	betas = (0.9, 0.999)
	
	random_seed = None
	#############################################
	
	# creating environment
	state_dim = 15*2 # 15 tanks with (x,y)
	action_dim = 4

	print(
		"state_dim:", state_dim, "\n"
		"action_dim:", action_dim, "\n"
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

	def parse_observations(observations: dict, kill_count: int, death_count: int) -> (list, int, bool):
		""" 
		parse observation dict into desirable data structures
		"""
		if not observations:
			print("no observations.. there is no one around me?")
			observations = {
				'deathCount': kill_count, 
				'killCount': death_count, 
				'radarScan': []
			}
		

		# grab the 15 closest tanks
		tank_locations = [ (float(tank_dic['x']), float(tank_dic['y']) ) for tank_dic in observations['radarScan']]
		tank_locations = sorted(tank_locations, key=parse_distances)

		# List of the closest tank coordinates
		ret_state = np.zeros((state_dim//2,2))

		if len(tank_locations) > 0:
			ret_state[:min(len(tank_locations), state_dim)] = tank_locations[:state_dim]

		print("I can see:", len(tank_locations), "tanks")
		# count of the kills
		ret_reward = observations['killCount']
		
		# have we died?
		ret_done = False if observations['deathCount'] == 0 else True

		return ret_state, ret_reward, ret_done

	# training loop
	for i_episode in range(1, max_episodes+1):
		try:
			env = RlWorldClient("129.127.147.237",1337)
			state = env.read_observation_dict()

			kill_count = 0
			death_count = 0
			for t in range(max_timesteps):
				time_step += 1
				start_time = time.time()

				# Run old policy
				state, current_kills, done = parse_observations(env.read_observation_dict(), kill_count, death_count)
				action = ppo.select_action(state.reshape(1, -1), memory).tolist()


				# calculate rewards
				reward_kills = current_kills - kill_count
				tmp_distances = (parse_distances(tank_loc) for tank_loc in state)
				reward_distance = sum((10-dist if dist > 0 else 0 for dist in tmp_distances))

				reward = sum([
					2    * reward_kills,
					0.01 * reward_distance
				])
				print("reward:", reward)
				kill_count += reward_kills

				# print(
				# 	"action:", action, "\n",
				# 	"state:", state, "\n", 
				# 	"current_kills:", current_kills, "\n", 
				# 	"done:", done, "\n",
				# 	"reward:", reward, "\n",
				# 	"kill_count:", kill_count, "\n"
				# 	)

				# Apply action to dict
				action_dict = {
					"name": f"Adrian_PPO_ep:{i_episode}_id:{id}",
					"colour": "#7017a1",
					"moveForwardBack": action[0],
					"moveRightLeft": action[1],
					"turnRightLeft": action[2],
					"fire": True,#action[3] > 0,
				}

				env.send_action_dict(action_dict)

				# Saving reward and is_terminals:
				memory.rewards.append(reward)
				memory.is_terminals.append(done)

				running_reward += reward

				# update if its time
				# print(time_step, update_timestep)
				print("I have:", len(memory.states), "memory")
				if time_step % update_timestep == 0:
					ppo.update(memory)
					memory.clear_memory()
					time_step = 0

				if done:
					break

				avg_length += t
				
				# sleep 
				end_time = time.time()
				time_diff = end_time-start_time
				time.sleep(0 if time_diff > tick_time else tick_time-time_diff)
		except Exception as e:
			print(e)
			time.sleep(1)

		
			
		# save every 500 episodes
		if i_episode % 500 == 0:
			torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format("test"))
			
		# logging
		# if i_episode % log_interval == 0:
		avg_length = int(avg_length/log_interval)
		running_reward = int((running_reward/log_interval))
		
		print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
		running_reward = 0
		avg_length = 0
			
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