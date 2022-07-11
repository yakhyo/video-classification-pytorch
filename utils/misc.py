import os
import errno

import torch
import torch.distributed as dist


class AverageMeter:
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def reduce_tensor(tensor, n):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= n
	return rt


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.inference_mode():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target[None])

		res = []
		for k in topk:
			correct_k = correct[:k].flatten().sum(dtype=torch.float32)
			res.append(correct_k * (100.0 / batch_size))
		return res


def mkdir(path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


def setup_for_distributed(is_master):
	"""This function disables printing when not in master process"""
	import builtins as __builtin__

	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		force = kwargs.pop("force", False)
		if is_master or force:
			builtin_print(*args, **kwargs)

	__builtin__.print = print


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


def is_main_process():
	return get_rank() == 0


def save_on_master(*args, **kwargs):
	if is_main_process():
		torch.save(*args, **kwargs)


def init_distributed_mode(args):
	if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
		args.local_rank = int(os.environ["RANK"])
		args.world_size = int(os.environ["WORLD_SIZE"])
		args.gpu = int(os.environ["LOCAL_RANK"])
	elif "SLURM_PROCID" in os.environ:
		args.local_rank = int(os.environ["SLURM_PROCID"])
		args.gpu = args.local_rank % torch.cuda.device_count()
	elif hasattr(args, "rank"):
		pass
	else:
		print("Not using distributed mode")
		args.distributed = False
		return

	args.distributed = True
	torch.cuda.set_device(args.gpu)
	args.dist_backend = "nccl"
	print(f"| distributed init (rank {args.local_rank}): {args.dist_url}", flush=True)
	torch.distributed.init_process_group(
		backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank
	)
	torch.distributed.barrier()
	setup_for_distributed(args.local_rank == 0)
