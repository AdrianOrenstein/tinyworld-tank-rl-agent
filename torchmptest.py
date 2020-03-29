import numpy as np
import time

import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Producer():
	def __init__(self, worker_id, processing_queue):
		super(Producer, self).__init__()
		self.id = worker_id
		self.processing_queue = processing_queue 
		self.inbox_queue = mp.Queue() 
		self.experience_queue = mp.Queue() 
		
		self.process = mp.Process(
			target=self.run
		)

	def run(self):
		"""
		A producer that pops numbers off the inputqueue, squares them and puts the result on resultqueue
		"""
		while True:
			state = np.random.rand(99, 1) # get observation from server

			self.processing_queue.put({
				'id': self.id, 
				'state': state,
				})


			action = self.inbox_queue.get()

			# send action to server
			
			# self.experience_queue.put({
			# 	'id': id, 
			# 	'state': state, 
			# 	'action': action
			# })

			# if None observation then shutdown
			if action is None:
				# print("Process:", id, "has shutdown")
				return

if __name__ == "__main__":
	mp.set_start_method('spawn')

	num_producers = 32
	consumer_batch_size = 32

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
					)
		
	def forward(self, x):
		raise NotImplementedError

	model = Net(99, 3)
	model.to(device)

	producers = {}
	processing_queue = mp.Queue() 

	# create producers as worker objects
	for producer_id in range(num_producers):
		producers[producer_id] = Producer(
			producer_id, 
			processing_queue
			)

	# start prducers
	for worker_id, worker in producers.items():
		worker.process.start()
	print("finished making producer(s)")

	experience = []
	jobs_done = 0

	model.eval()
	
	start_time = time.time()
	while jobs_done <= 8192:
		jobs = []
		while (not processing_queue.empty() or len(jobs) == 0) and len(jobs) < consumer_batch_size:
			jobs.append(processing_queue.get())

		if jobs:
			with torch.no_grad():
				batch = torch.FloatTensor(np.stack([dic['state'].reshape(1, -1) for dic in jobs])).to(device)
				inference = model.network(batch)
				inference = inference.cpu().data.numpy()

				for w_id, result in zip((j['id'] for j in jobs), np.rollaxis(inference, 0)):
					producers[w_id].inbox_queue.put(result)
					# experience.append(producers[w_id].experience_queue.get())

				jobs_done += len(jobs)
	
	print(f"Full experience received in: {time.time() - start_time:.2f}s")

	print(f"Closing {len(producers)} producers with {len(experience)} samples")
	for worker_id, worker in producers.items():
		worker.inbox_queue.put(None)

	for worker_id, worker in producers.items():
		worker.process.join()

	print("All done")