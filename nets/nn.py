from typing import Tuple, Optional, Callable, List, Sequence, Type, Any, Union

import torch.nn as nn
from torch import Tensor

__all__ = [
	"VideoResNet",
	"r3d_18",
	"mc3_18",
	"r2plus1d_18",
]


class Conv3DSimple(nn.Conv3d):
	def __init__(
			self, in_planes: int, out_planes: int, midplanes: Optional[int] = None, stride: int = 1, padding: int = 1
	) -> None:
		super().__init__(
			in_channels=in_planes,
			out_channels=out_planes,
			kernel_size=(3, 3, 3),
			stride=stride,
			padding=padding,
			bias=False,
		)

	@staticmethod
	def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
		return stride, stride, stride


class Conv2Plus1D(nn.Sequential):
	def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding: int = 1) -> None:
		super().__init__(
			nn.Conv3d(
				in_planes,
				midplanes,
				kernel_size=(1, 3, 3),
				stride=(1, stride, stride),
				padding=(0, padding, padding),
				bias=False,
			),
			nn.BatchNorm3d(midplanes),
			nn.ReLU(inplace=True),
			nn.Conv3d(
				midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
			),
		)

	@staticmethod
	def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
		return stride, stride, stride


class Conv3DNoTemporal(nn.Conv3d):
	def __init__(
			self, in_planes: int, out_planes: int, midplanes: Optional[int] = None, stride: int = 1, padding: int = 1
	) -> None:
		super().__init__(
			in_channels=in_planes,
			out_channels=out_planes,
			kernel_size=(1, 3, 3),
			stride=(1, stride, stride),
			padding=(0, padding, padding),
			bias=False,
		)

	@staticmethod
	def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
		return 1, stride, stride


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(
			self,
			inplanes: int,
			planes: int,
			conv_builder: Callable[..., nn.Module],
			stride: int = 1,
			downsample: Optional[nn.Module] = None,
	) -> None:
		midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

		super().__init__()
		self.conv1 = nn.Sequential(
			conv_builder(inplanes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
		)
		self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x: Tensor) -> Tensor:
		residual = x

		out = self.conv1(x)
		out = self.conv2(out)
		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class BasicStem(nn.Sequential):
	"""The default conv-batchnorm-relu stem"""

	def __init__(self) -> None:
		super().__init__(
			nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
			nn.BatchNorm3d(64),
			nn.ReLU(inplace=True),
		)


class R2Plus1dStem(nn.Sequential):
	"""R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

	def __init__(self) -> None:
		super().__init__(
			nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
			nn.BatchNorm3d(45),
			nn.ReLU(inplace=True),
			nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
			nn.BatchNorm3d(64),
			nn.ReLU(inplace=True),
		)


class VideoResNet(nn.Module):
	def __init__(
			self,
			block: Type[BasicBlock],
			conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
			layers: List[int],
			stem: Callable[..., nn.Module],
			num_classes: int = 400,
			zero_init_residual: bool = False,
	) -> None:

		super().__init__()
		self.inplanes = 64

		self.stem = stem()

		self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		# init weights
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


	def forward(self, x: Tensor) -> Tensor:
		x = self.stem(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		# Flatten the layer to fc
		x = x.flatten(1)
		x = self.fc(x)

		return x

	def _make_layer(
			self,
			block: Type[BasicBlock],
			conv_builder: Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]],
			planes: int,
			blocks: int,
			stride: int = 1,
	) -> nn.Sequential:
		downsample = None

		if stride != 1 or self.inplanes != planes * block.expansion:
			ds_stride = conv_builder.get_downsample_stride(stride)
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
				nn.BatchNorm3d(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, conv_builder))

		return nn.Sequential(*layers)


def _video_resnet(
		block: Type[BasicBlock],
		conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
		layers: List[int],
		stem: Callable[..., nn.Module],
		**kwargs: Any,
) -> VideoResNet:
	model = VideoResNet(block, conv_makers, layers, stem, **kwargs)

	return model


def r3d_18(**kwargs: Any) -> VideoResNet:
	return _video_resnet(
		BasicBlock,
		[Conv3DSimple] * 4,
		[2, 2, 2, 2],
		BasicStem,
		**kwargs,
	)


def mc3_18(**kwargs: Any) -> VideoResNet:
	return _video_resnet(
		BasicBlock,
		[Conv3DSimple] + [Conv3DNoTemporal] * 3,  # type: ignore[list-item]
		[2, 2, 2, 2],
		BasicStem,
		**kwargs,
	)


def r2plus1d_18(**kwargs) -> VideoResNet:
	"""Construct 18 layer deep R(2+1)D network as in
	.. betastatus:: video module
	Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.
	Args:
		weights (:class:`~torchvision.models.video.R2Plus1D_18_Weights`, optional): The
			pretrained weights to use. See
			:class:`~torchvision.models.video.R2Plus1D_18_Weights`
			below for more details, and possible values. By default, no
			pre-trained weights are used.
		progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
		**kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
			Please refer to the `source code
			<https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
			for more details about this class.
	.. autoclass:: torchvision.models.video.R2Plus1D_18_Weights
		:members:
	"""

	return _video_resnet(
		BasicBlock,
		[Conv2Plus1D] * 4,
		[2, 2, 2, 2],
		R2Plus1dStem,
		**kwargs,
	)


def profile(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
	model = r2plus1d_18()
	print(profile(model))
