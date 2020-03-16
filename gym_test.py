import numpy as np
from rlworldclient import RlWorldClient
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from gym.spaces import Discrete, Box

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search

tf = try_import_tf()


class AIMLTank(gym.Env):

	def __init__(self, config):
		self.env = RlWorldClient("129.127.147.237",1337)
		self.state_dim = config['state_dimensions']
		self.id = random.randint(0,1000)

		self.action_space = Box(-1, 1, shape=(4, ), dtype=np.float32)
		self.observation_space = Box(-10, 10, shape=(self.state_dim,2), dtype=np.float32)

		self.state, self.kill_count, self.death_count = self.parse_observations(self.env.read_observation_dict(), self.kill_count, self.death_count)

		self.mm_clip = lambda x, l, u: max(l, min(u, x))

	def reset(self):
		self.kill_count = 0
		self.death_count = 0
		self.state, _, _ = self.parse_observations(self.env.read_observation_dict(), self.kill_count, self.death_count)

	def parse_observations(self, observations: dict, kill_count: int, death_count: int) -> (list, int, bool):
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
				'deathCount': kill_count, 
				'killCount': death_count, 
				'radarScan': []
			}
		

		# grab the 15 closest tanks
		tank_locations = [ (float(tank_dic['x']), float(tank_dic['y']) ) for tank_dic in observations['radarScan']]
		tank_locations = sorted(tank_locations, key=parse_distances)

		# List of the closest tank coordinates
		ret_state = np.zeros((self.state_dim//2,2))

		for i, loc in enumerate(tank_locations):
			ret_state[i] = loc

		# count of the kills
		ret_reward = observations['killCount']
		
		# have we died?
		ret_episode_over = False if observations['deathCount'] == 0 else True

		return ret_state, ret_reward, ret_episode_over

	def step(self, action):
		self._take_action(action)

		# Run old policy
		state, current_kills, episode_over = self.parse_observations(self.env.read_observation_dict(), self.kill_count, self.death_count)

		# calculate rewards
		reward_kills = current_kills - kill_count
		tmp_distances = (parse_distances(tank_loc) for tank_loc in state)
		reward_distance = sum((self.mm_clip(1/dist, 0, 1) if dist>0 else 0 for dist in tmp_distances))

		reward = sum([
			2    * reward_kills,
			0.01 * reward_distance
		])
		self.kill_count += reward_kills

		return state, reward, episode_over, {
			'reward_kills': 2*reward_kills, 
			'reward_dist': 0.01*reward_distance
		}


	def _take_action(self, action):
		# Apply action to dict
		action_dict = {
			"name": f"swarm_ep:{i_episode}_id:{self.id}",
			"colour": "#7017a1",
			"moveForwardBack": self.mm_clip(action[0], -1, 1),
			"moveRightLeft": self.mm_clip(action[1], -1, 1),
			"turnRightLeft": self.mm_clip(action[2], -1, 1),
			"fire": True, # action[3] > 0,
		}

		env.send_action_dict(action_dict)


class CustomModel(TFModelV2):
	"""Example of a custom model that just delegates to a fc-net."""

	def __init__(self, obs_space, action_space, num_outputs, model_config,
				 name):
		super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
										  model_config, name)
		self.model = FullyConnectedNetwork(obs_space, action_space,
										   num_outputs, model_config, name)
		self.register_variables(self.model.variables())

	def forward(self, input_dict, state, seq_lens):
		return self.model.forward(input_dict, state, seq_lens)

	def value_function(self):
		return self.model.value_function()


if __name__ == "__main__":
	# Can also register the env creator function explicitly with:
	# register_env("corridor", lambda config: SimpleCorridor(config))
	ray.init()
	ModelCatalog.register_custom_model("my_model", CustomModel)
	tune.run(
		"PPO",
		stop={
			"timesteps_total": 100,
		},
		config={
			"env": AIMLTank,  # or "corridor" if registered above
			"model": {
				"custom_model": "my_model",
			},
			"vf_share_layers": True,
			"lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
			"num_workers": 1,  # parallelism
			"env_config": {
				"state_dimensions": 15,
			}
		},
	)