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


while True:
    try:
        client = RlWorldClient("10.90.159.11", 1337)
        while True:
            # print("read")
            # print(client.read_observation_dict())
            # print("send")

            action_dict = {
                "name": "bot_dumb_" + str(id),
                "colour": "#000000",
                "moveForwardBack":random.uniform(-1.0,1.0),
                "moveRightLeft": 0, #random.uniform(-1.0,1.0),
                "turnRightLeft": random.uniform(-1.0,1.0),
                "fire": True if random.uniform(-1.0,1.0) > 0 else False,
            }

            client.send_action_dict(action_dict)
            # print("sleep")
            time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)