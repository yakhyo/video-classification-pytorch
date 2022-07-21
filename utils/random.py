import random
import numpy as np
import torch


def random_seed(seed=42):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
