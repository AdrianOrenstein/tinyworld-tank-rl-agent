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


import torch.multiprocessing as mp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, action_std):
#         super(ActorCritic, self).__init__()
#         # action mean range -1 to 1
#         self.hidden_size = 64
#         self.actor =  nn.Sequential(
#                 nn.Linear(state_dim, self.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_size, self.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_size, self.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_size, self.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_size, action_dim),
#                 nn.Tanh(),
#                 ).to(device)

#         self.actor.share_memory()

#         # critic
#         self.critic = nn.Sequential(
#                 nn.Linear(state_dim, self.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_size, self.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_size, 1)
#                 ).to(device)
#         self.critic.share_memory()

#         self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
		
#     def forward(self):
#         raise NotImplementedError
	
#     def act(self, state):
#         action_mean = self.actor(state)
#         cov_mat = torch.diag(self.action_var).to(device)
		
#         dist = MultivariateNormal(action_mean, cov_mat)
#         action = dist.sample()
#         action_logprob = dist.log_prob(action)
		
#         return action.detach(), action_logprob.detach()
	
#     def evaluate(self, state, action):   
#         action_mean = self.actor(state)
		
#         action_var = self.action_var.expand_as(action_mean)
#         cov_mat = torch.diag_embed(action_var).to(device)
		
#         dist = MultivariateNormal(action_mean, cov_mat)
		
#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
#         state_value = self.critic(state)
		
#         return action_logprobs, torch.squeeze(state_value), dist_entropy

# class Producer:
#     def __init__(self, worker_id, send_for_processing_queue, inbox_queue, experience_queue, hparams):
#         self.id = worker_id
#         self.send_for_processing_queue = send_for_processing_queue
#         self.inbox_queue = inbox_queue
#         self.experience_queue = experience_queue

#         self.global_obs = dict()
#         self.hparams = hparams
#         self.state = RingBuffer(capacity=self.hparams['state_dim'], dtype=np.float32)

#     # Construct data_loader, optimizer, etc.
#     state = torch.tensor(np.zeros((99,1)))
#     return model.act(state)

# if __name__ == '__main__':
#     num_processes = 4
#     state_dim = 99
#     action_dim = 3
#     action_std = 0.5
#     model = ActorCritic(state_dim, action_dim, action_std)

#     mp.set_start_method('spawn')

#     queue = mp.Queue()
#     res = mp.spawn(
#         fn=run_worker, 
#         args=(ppo, hparams, queue),
#         nprocs=num_workers
#     )

#     print(processes)

def producer(id, processing_queue, inbox_queue, experience_queue):
	"""
	A producer that pops numbers off the inputqueue, squares them and puts the result on resultqueue
	"""
	while True:
		# get observation from server
		state = np.random.rand(99, 1)

		processing_queue.put({
			'id': id, 
			'state': state,
			})


		action = inbox_queue.get()

		# send action to server

		
		experience_queue.put({
			'id': id, 
			'state': state, 
			'action': action
		})

		# if None observation then shutdown
		if action is None:
			return


# def consumer(model, processing_queue, producers):
# 	"""
# 	A consumer that pops results off the resultqueue and prints them to screen
# 	"""
# 	while True:
# 		job = processing_queue.get()
# 		print(processing_queue.qsize())
# 		
# 		print(len(producers), "".join((f"{worker_id}:\t{worker.inbox_queue.qsize()}\t{worker.experience_queue.qsize()}"for worker_id, worker in producers.items())))
# 		# if None observation then shutdown
# 		if job is None:
# 			return

# 		# run inference on state
# 		print(job)
# 		print(job['id'], job['state'])
# 		inference = model.network(torch.tensor(job['state']).to(device))
# 		producers[job['id']].inbox_queue.put(inference)
# 		print("Consumed job from: ", job['id'])
# 		del job, inference


class Worker():
	def __init__(self, worker_id, processing_queue):
		super(Worker, self).__init__()
		self.id = worker_id
		self.processing_queue = processing_queue 
		self.inbox_queue = mp.Queue() 
		self.experience_queue = mp.Queue() 
		
		self.process = mp.Process(
			target=producer, 
			args=(
				worker_id, 
				self.processing_queue, 
				self.inbox_queue, 
				self.experience_queue
			)
		)

if __name__ == "__main__":
	mp.set_start_method('spawn')

	num_producers = 8

	# Generate input
	# inputqueue = Queue()
	# for i in range(100):
	# 	inputqueue.put(i % 10)
	# for _ in range(num_producers):
	# 	inputqueue.put(None)  # Ensures that producers terminate

	class Net(nn.Module):
		def __init__(self, state_dim, action_dim):
			super(Net, self).__init__()
			# action mean range -1 to 1
			self.hidden_size = 64
			self.network =  nn.Sequential(
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
		
	def forward(self, x):
		raise NotImplementedError

	model = Net(99, 3)
	model.to(device)

	producers = dict()
	processing_queue = mp.Queue() 

	# create producers as worker objects
	for producer_id in range(num_producers):
		producers[producer_id] = Worker(
			producer_id, 
			processing_queue
			)

	# consumer = mp.Process(target=consumer, args=(model, processing_queue, producers))

	# start prducers
	for worker_id, worker in producers.items():
		print(worker.process)
		worker.process.start()
	# consumer.start()

	print("finished making producer(s)")

	jobs_done = 0

	model.eval()
	with torch.no_grad():
		while jobs_done < 256:
			# https://docs.python.org/3.4/library/multiprocessing.html?highlight=process#multiprocessing.Queue
			job = processing_queue.get()
			
			# print(len(producers), "".join((f"{worker_id}:\t{worker.inbox_queue.qsize()}\t{worker.experience_queue.qsize()}"for worker_id, worker in producers.items())))

			# run inference on state
			# print(job)
			# print(job['id'], "data:", job['state'].shape)

			inference = model.network(torch.FloatTensor(job['state'].reshape(1, -1)).to(device))
			inference = inference.cpu().data.numpy().flatten()

			producers[job['id']].inbox_queue.put(inference)
			jobs_done += 1
			# print("Consumed job from: ", job['id'])
			# print("\n\n\n")

	experience = []

	for worker_id, worker in producers.items():
		while not worker.experience_queue.empty():
			experience.append(worker.experience_queue.get())

	print("Experience gained:", len(experience))

	# Wait for producers to finish
	for worker_id, worker in producers.items():
		worker.inbox_queue.put(None)
		worker.process.join()

	# Wait for consumer to finish
	processing_queue.put(None)

	print("All done")