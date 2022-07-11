import os
import time
import datetime
import warnings

from utils import misc, presets
from utils.misc import AverageMeter, reduce_tensor  # test
import torch.utils.data
import torchvision

# import torchvision.datasets.video_utils

from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import DistributedSampler, UniformClipSampler, RandomClipSampler

warnings.filterwarnings('ignore')


def train_one_epoch(args, model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, scaler=None):
	model.train()
	last_idx = len(data_loader) - 1

	time_logger = AverageMeter()  # img/s
	loss_logger = AverageMeter()  # loss
	top1_logger = AverageMeter()  # top1 accuracy
	top5_logger = AverageMeter()  # top5 accuracy

	for batch_idx, (video, target) in enumerate(data_loader):
		last_batch = batch_idx == last_idx
		batch_size = video.shape[0]
		start_time = time.time()
		video, target = video.to(device), target.to(device)
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			output = model(video)
			loss = criterion(output, target)

		optimizer.zero_grad()

		if scaler is not None:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		torch.cuda.synchronize()

		if last_batch or batch_idx % args.print_freq == 0:
			lrl = [param_group['lr'] for param_group in optimizer.param_groups]
			lr = sum(lrl) / len(lrl)
			acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

			if args.distributed:
				loss = reduce_tensor(loss.data, args.world_size)
				acc1 = reduce_tensor(acc1, args.world_size)
				acc5 = reduce_tensor(acc5, args.world_size)
			else:
				loss = loss.data

			loss_logger.update(loss.item(), n=batch_size)
			top1_logger.update(acc1.item(), n=batch_size)
			top5_logger.update(acc5.item(), n=batch_size)
			time_logger.update(batch_size / (time.time() - start_time))

			if args.local_rank == 0:
				print(
					"Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
					"Loss: {loss.val:>6.4f} ({loss.avg:>6.4f})  "
					"Acc@1: {acc1.val:>6.4f} ({acc1.avg:>6.4f}) "
					"Acc@5: {acc5.val:>6.4f} ({acc5.avg:>6.4f}) "
					"LR: {lr:.3e}  "
					"Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
						epoch, batch_idx, len(data_loader),
						100. * batch_idx / last_idx,
						loss=loss_logger,
						acc1=top1_logger,
						acc5=top5_logger,
						lr=lr,
						data_time=time_logger))

		lr_scheduler.step()


def evaluate(args, model, criterion, data_loader, device):
	model.eval()

	time_logger = AverageMeter()  # img/s
	loss_logger = AverageMeter()  # loss
	top1_logger = AverageMeter()  # top1 accuracy
	top5_logger = AverageMeter()  # top5 accuracy

	header = "Test:"

	end = time.time()
	last_idx = len(data_loader) - 1

	with torch.inference_mode():
		for batch_idx, (video, target) in enumerate(data_loader):
			last_batch = batch_idx == last_idx
			video = video.to(device, non_blocking=True)
			target = target.to(device, non_blocking=True)

			output = model(video)
			loss = criterion(output, target)

			torch.cuda.synchronize()
			batch_size = video.shape[0]

			if last_batch or batch_idx % args.print_freq == 0:

				acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

				if args.distributed:
					loss = reduce_tensor(loss.data, args.world_size)
					acc1 = reduce_tensor(acc1, args.world_size)
					acc5 = reduce_tensor(acc5, args.world_size)
				else:
					loss = loss.data

				loss_logger.update(loss.item(), n=batch_size)
				top1_logger.update(acc1.item(), n=batch_size)
				top5_logger.update(acc5.item(), n=batch_size)
				time_logger.update(batch_size / (time.time() - end))

				if args.local_rank == 0:
					print(
						'{0}: [{1:>4d}/{2}]  '
						'Time: {batch_time.val:>4.3f} ({batch_time.avg:>4.3f})  '
						'Loss: {loss.val:>4.4f} ({loss.avg:>6.4f})  '
						'Acc@1: {top1.val:>4.4f} ({top1.avg:>4.4f})  '
						'Acc@5: {top5.val:>4.4f} ({top5.avg:>4.4f})'.format(
							header, batch_idx, last_idx,
							batch_time=time_logger,
							loss=loss_logger,
							top1=top1_logger,
							top5=top5_logger)
					)

	print(f"{header} Loss: {loss_logger.avg:.3f} Acc@1 {top1_logger.avg:.3f} Acc@5 {top5_logger.avg:.3f}")
	return loss_logger.avg, top1_logger.avg, top5_logger.avg


def _get_cache_path(filepath):
	import hashlib

	h = hashlib.sha1(filepath.encode()).hexdigest()
	cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
	cache_path = os.path.expanduser(cache_path)
	return cache_path


def collate_fn(batch):
	# remove audio from the batch
	batch = [(d[0], d[2]) for d in batch]
	return default_collate(batch)


def main(args):
	if args.output_dir:
		misc.mkdir(args.output_dir)

	misc.init_distributed_mode(args)
	print(args)

	device = torch.device(args.device)

	if args.use_deterministic_algorithms:
		torch.backends.cudnn.benchmark = False
		torch.use_deterministic_algorithms(True)
	else:
		torch.backends.cudnn.benchmark = True

	# Data loading code
	print("Loading data")
	traindir = os.path.join(args.data_path, "train")
	valdir = os.path.join(args.data_path, "val")

	print("Loading training data")
	st = time.time()
	cache_path = _get_cache_path(traindir)
	transform_train = presets.VideoClassificationPresetTrain(crop_size=(112, 112), resize_size=(128, 171))

	if args.cache_dataset and os.path.exists(cache_path):
		print(f"Loading dataset_train from {cache_path}")
		dataset, _ = torch.load(cache_path)
		dataset.transform = transform_train
	else:
		if args.distributed:
			print("It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster")
		dataset = torchvision.datasets.UCF101(
			args.data_path,
			annotation_path='../../Datasets/UCF-101/annotations',
			frames_per_clip=args.clip_len,
			train=True,
			step_between_clips=1,
			transform=transform_train,
			frame_rate=args.frame_rate,
		)
		if args.cache_dataset:
			print(f"Saving dataset_train to {cache_path}")
			misc.mkdir(os.path.dirname(cache_path))
			misc.save_on_master((dataset, traindir), cache_path)

	print("Took", time.time() - st)

	print("Loading validation data")
	cache_path = _get_cache_path(valdir)

	if args.weights and args.test_only:
		weights = torchvision.models.get_weight(args.weights)
		transform_test = weights.transforms()
	else:
		transform_test = presets.VideoClassificationPresetEval(crop_size=(112, 112), resize_size=(128, 171))

	if args.cache_dataset and os.path.exists(cache_path):
		print(f"Loading dataset_test from {cache_path}")
		dataset_test, _ = torch.load(cache_path)
		dataset_test.transform = transform_test
	else:
		if args.distributed:
			print("It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster")
		dataset_test = torchvision.datasets.UCF101(
			args.data_path,
			annotation_path='../../Datasets/UCF-101/annotations',
			frames_per_clip=args.clip_len,
			train=False,
			step_between_clips=1,
			transform=transform_test,
			frame_rate=args.frame_rate,
		)
		if args.cache_dataset:
			print(f"Saving dataset_test to {cache_path}")
			misc.mkdir(os.path.dirname(cache_path))
			misc.save_on_master((dataset_test, valdir), cache_path)

	print("Creating data loaders")
	train_sampler = RandomClipSampler(dataset.video_clips, args.clips_per_video)
	test_sampler = UniformClipSampler(dataset_test.video_clips, args.clips_per_video)
	if args.distributed:
		train_sampler = DistributedSampler(train_sampler)
		test_sampler = DistributedSampler(test_sampler, shuffle=False)

	data_loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=args.batch_size,
		sampler=train_sampler,
		num_workers=args.workers,
		pin_memory=True,
		collate_fn=collate_fn,
	)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test,
		batch_size=args.batch_size,
		sampler=test_sampler,
		num_workers=args.workers,
		pin_memory=True,
		collate_fn=collate_fn,
	)

	print("Creating model")
	model = torchvision.models.video.__dict__[args.model](weights=args.weights)
	model.to(device)
	if args.distributed and args.sync_bn:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

	criterion = nn.CrossEntropyLoss()

	lr = args.lr * args.world_size
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
	scaler = torch.cuda.amp.GradScaler() if args.amp else None

	# convert scheduler to be per iteration, not per epoch, for warmup that lasts
	# between different epochs
	iters_per_epoch = len(data_loader)
	lr_milestones = [iters_per_epoch * (m - args.lr_warmup_epochs) for m in args.lr_milestones]
	main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

	if args.lr_warmup_epochs > 0:
		warmup_iters = iters_per_epoch * args.lr_warmup_epochs
		args.lr_warmup_method = args.lr_warmup_method.lower()
		if args.lr_warmup_method == "linear":
			warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
				optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
			)
		elif args.lr_warmup_method == "constant":
			warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
				optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
			)
		else:
			raise RuntimeError(
				f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
			)

		lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
			optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
		)
	else:
		lr_scheduler = main_lr_scheduler

	model_without_ddp = model
	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		model_without_ddp = model.module

	if args.resume:
		checkpoint = torch.load(args.resume, map_location="cpu")
		model_without_ddp.load_state_dict(checkpoint["model"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
		args.start_epoch = checkpoint["epoch"] + 1
		if args.amp:
			scaler.load_state_dict(checkpoint["scaler"])

	if args.test_only:
		# We disable the cudnn benchmarking because it can noticeably affect the accuracy
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		evaluate(model, criterion, data_loader_test, device=device)
		return

	print("Start training")
	start_time = time.time()
	best = 0
	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)
		train_one_epoch(args, model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, scaler)
		loss, acc1, acc5 = evaluate(args, model, criterion, data_loader_test, device=device)
		if args.output_dir:
			checkpoint = {
				"model": model_without_ddp.state_dict(),
				"optimizer": optimizer.state_dict(),
				"lr_scheduler": lr_scheduler.state_dict(),
				"epoch": epoch,
				"args": args,
			}
			if args.amp:
				checkpoint["scaler"] = scaler.state_dict()

			misc.save_on_master(checkpoint, os.path.join(args.output_dir, "last.pth"))
			if best < acc1:
				misc.save_on_master(checkpoint, os.path.join(args.output_dir, "best.pth"))

			best = max(best, acc1)

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print(f"Training time {total_time_str}")


def parse_args():
	import argparse

	parser = argparse.ArgumentParser(description="PyTorch Video Classification Training")

	parser.add_argument("--data-path", default="../../Datasets/UCF-101/videos/", type=str, help="dataset path")
	parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
	parser.add_argument("--clip-len", default=16, type=int, metavar="N", help="number of frames per clip")
	parser.add_argument("--frame-rate", default=15, type=int, metavar="N", help="the frame rate")
	parser.add_argument(
		"--clips-per-video", default=5, type=int, metavar="N", help="maximum number of clips per video to consider"
	)
	parser.add_argument(
		"-b", "--batch-size", default=48, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
	)
	parser.add_argument("--epochs", default=2, type=int, metavar="N", help="number of total epochs to run")
	parser.add_argument(
		"-j", "--workers", default=24, type=int, metavar="N", help="number of data loading workers (default: 10)"
	)
	parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
	parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
	parser.add_argument(
		"--wd",
		"--weight-decay",
		default=1e-4,
		type=float,
		metavar="W",
		help="weight decay (default: 1e-4)",
		dest="weight_decay",
	)
	parser.add_argument("--lr-milestones", nargs="+", default=[20, 30, 40], type=int, help="decrease lr on milestones")
	parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
	parser.add_argument("--lr-warmup-epochs", default=10, type=int, help="the number of epochs to warmup (default: 10)")
	parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
	parser.add_argument("--lr-warmup-decay", default=0.001, type=float, help="the decay for lr")
	parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
	parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
	parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
	parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
	parser.add_argument(
		"--cache-dataset",
		default=False,
		dest="cache_dataset",
		help="Cache the datasets for quicker initialization. It also serializes the transforms",
		action="store_true",
	)
	parser.add_argument(
		"--sync-bn",
		dest="sync_bn",
		help="Use sync batch norm",
		action="store_true",
	)
	parser.add_argument(
		"--test-only",
		dest="test_only",
		help="Only test the model",
		action="store_true",
	)
	parser.add_argument(
		"--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
	)

	# distributed training parameters
	parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
	parser.add_argument("--local-rank", default=0, type=int, help="number of distributed processes")
	parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

	parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

	# Mixed precision training parameters
	parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

	args = parser.parse_args()

	return args


if __name__ == "__main__":
	args = parse_args()
	main(args)
