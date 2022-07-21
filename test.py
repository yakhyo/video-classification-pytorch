import os
import time
import datetime
import warnings

from sklearn.metrics import precision_score, recall_score

from nets.nn import r3d_18, mc3_18, r2plus1d_18
from utils import misc, presets
from utils.misc import AverageMeter

import torch
import torchvision
import torch.utils.data

from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import UniformClipSampler

warnings.filterwarnings("ignore")


def validate(args, model, criterion, data_loader, device):
	model.eval()

	time_logger = AverageMeter()  # img/s
	loss_logger = AverageMeter()  # loss
	top1_logger = AverageMeter()  # top1 accuracy
	top5_logger = AverageMeter()  # top5 accuracy

	header = "Test"

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

				loss = loss.data

				loss_logger.update(loss.item(), n=batch_size)
				top1_logger.update(acc1.item(), n=batch_size)
				top5_logger.update(acc5.item(), n=batch_size)
				time_logger.update(batch_size / (time.time() - end))

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
	cache_path = os.path.join("~", "Datasets", "UCF-101", "cache", h[:10] + ".pt")
	cache_path = os.path.expanduser(cache_path)
	return cache_path


def collate_fn(batch):
	# remove audio from the batch
	batch = [(d[0], d[2]) for d in batch]
	return default_collate(batch)


def main(args):
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Data loading code
	print("Loading validation data")
	valdir = os.path.join(args.data_path, "val")
	cache_path = _get_cache_path(valdir)
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
			annotation_path=args.annotations,
			frames_per_clip=args.clip_len,
			train=False,
			num_workers=os.cpu_count(),
			step_between_clips=1,
			transform=transform_test,
			frame_rate=args.frame_rate,
		)
		if args.cache_dataset:
			print(f"Saving dataset_test to {cache_path}")
			misc.mkdir(os.path.dirname(cache_path))
			misc.save_on_master((dataset_test, valdir), cache_path)

	print("Creating data loaders")
	test_sampler = UniformClipSampler(dataset_test.video_clips, args.clips_per_video)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test,
		batch_size=args.batch_size,
		sampler=test_sampler,
		num_workers=args.workers,
		pin_memory=True,
		collate_fn=collate_fn,
	)

	print("Creating model")
	model = r2plus1d_18(num_classes=len(dataset_test.classes)).to(device)

	model.load_state_dict(torch.load(args.weights, map_location="cpu")["model"])
	print(f"Loaded weights from: {args.weights}")
	criterion = nn.CrossEntropyLoss()

	print("Start evaluation")
	start_time = time.time()

	validate(args, model, criterion, data_loader_test, device=device)

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))

	print(f"Evaluation time {total_time_str}")


def parse_args():
	import argparse

	parser = argparse.ArgumentParser(description="RacyVideo Validation")

	# Data params
	parser.add_argument("--data-path", default="../../Datasets/UCF-101/videos", type=str, help="dataset path")
	parser.add_argument("--annotations", default="../../Datasets/UCF-101/annotations", type=str, help="dataset path")
	parser.add_argument("--cache-dataset", default=True, action="store_true", help="for quicker initialization")

	# Video frame params
	parser.add_argument("--clip-len", default=16, type=int, metavar="N", help="number of frames per clip")
	parser.add_argument("--frame-rate", default=15, type=int, metavar="N", help="the frame rate")
	parser.add_argument("--clips-per-video", default=5, type=int, help="maximum number of clips per video to consider")

	# Weights file
	parser.add_argument("--weights", default="weights/best.pth", type=str, help="path to weight file")

	# DataLoader & print params
	parser.add_argument("--batch-size", default=24, type=int, help="images per gpu, total = $NGPU x batch_size")
	parser.add_argument("--workers", default=24, type=int, metavar="N", help="number of data loading workers")
	parser.add_argument("--print-freq", default=10, type=int, help="print frequency")

	args = parser.parse_args()

	return args


if __name__ == "__main__":
	arg_parse = parse_args()
	main(arg_parse)
