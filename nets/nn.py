from typing import Tuple, Optional, Callable, List, Sequence, Type, Any, Union

from torch import nn, Tensor

__all__ = [
	"VideoResNet",
	"r3d_18",
	"mc3_18",
	"r2plus1d_18",
]

__author__ = ['John', 'yakhyo9696@gmail.com']


class Conv3DSimple(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 1) -> None:
		super().__init__()
		self.conv3d = nn.Conv3d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=(3, 3, 3),
			stride=stride,
			padding=padding,
			bias=False,
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.conv3d(x)
		return x

	@staticmethod
	def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
		return stride, stride, stride


class Conv3DNoTemporal(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 1) -> None:
		super().__init__()
		self.conv3d = nn.Conv3d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=(1, 3, 3),
			stride=(1, stride, stride),
			padding=(0, padding, padding),
			bias=False,
		)

	def forward(self, x):
		x = self.conv3d(x)
		return x

	@staticmethod
	def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
		return 1, stride, stride


class Conv2Plus1D(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			mid_channels: int,
			stride: int = 1,
			padding: int = 1
	) -> None:
		super().__init__()

		self.stem = nn.Sequential(
			nn.Conv3d(
				in_channels,
				mid_channels,
				kernel_size=(1, 3, 3),
				stride=(1, stride, stride),
				padding=(0, padding, padding),
				bias=False,
			),
			nn.BatchNorm3d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv3d(
				mid_channels,
				out_channels,
				kernel_size=(3, 1, 1),
				stride=(stride, 1, 1),
				padding=(padding, 0, 0),
				bias=False
			),
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.stem(x)
		return x

	@staticmethod
	def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
		return stride, stride, stride


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(
			self,
			in_channels: int,
			channels: int,
			conv_builder: Callable[..., nn.Module],
			stride: int = 1,
			downsample: Optional[nn.Module] = None,
	) -> None:
		mid_channels = (in_channels * channels * 3 * 3 * 3) // (in_channels * 3 * 3 + 3 * channels)

		super().__init__()
		self.conv1 = nn.Sequential(
			conv_builder(in_channels, channels, mid_channels, stride), nn.BatchNorm3d(channels), nn.ReLU(inplace=True)
		)
		self.conv2 = nn.Sequential(conv_builder(channels, channels, mid_channels), nn.BatchNorm3d(channels))
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


class Conv3dBlock(nn.Module):
	"""The default conv-batchnorm-relu stem"""

	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, inplace=True) -> None:
		super().__init__()
		self.stem = nn.Sequential(
			nn.Conv3d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding,
				bias=bias,
			),
			nn.BatchNorm3d(num_features=out_channels),
			nn.ReLU(inplace=inplace),
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.stem(x)
		return x


BASIC_STEM_BLOCK = Conv3dBlock(
	in_channels=3,
	out_channels=64,
	kernel_size=(3, 7, 7),
	stride=(1, 2, 2),
	padding=(1, 3, 3),
	bias=False,
	inplace=True,
)


class R2Plus1dStem(nn.Module):
	"""R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

	def __init__(self) -> None:
		super().__init__()
		self.stem1 = Conv3dBlock(
			in_channels=3,
			out_channels=45,
			kernel_size=(1, 7, 7),
			stride=(1, 2, 2),
			padding=(0, 3, 3),
			bias=False,
			inplace=True,
		)

		self.stem2 = Conv3dBlock(
			in_channels=45,
			out_channels=64,
			kernel_size=(3, 1, 1),
			stride=(1, 1, 1),
			padding=(1, 0, 0),
			bias=False,
			inplace=True,
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.stem1(x)
		x = self.stem2(x)

		return x


class VideoResNet(nn.Module):
	def __init__(
			self,
			block: Type[BasicBlock],
			conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
			layers: List[int],
			stem: Callable[..., nn.Module],
			num_classes: int = 400,
	) -> None:
		"""Generic resnet video generator.
		Args:
			block (Type[BasicBlock]): resnet building block
			conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
				function for each layer
			layers (List[int]): number of blocks per layer
			stem (Callable[..., nn.Module]): module specifying the ResNet stem.
			num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
		"""
		super().__init__()
		self.in_channels = 64

		self.stem = stem

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
			channels: int,
			blocks: int,
			stride: int = 1,
	) -> nn.Sequential:
		downsample = None

		if stride != 1 or self.in_channels != channels * block.expansion:
			ds_stride = conv_builder.get_downsample_stride(stride)
			downsample = nn.Sequential(
				nn.Conv3d(self.in_channels, channels * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
				nn.BatchNorm3d(channels * block.expansion),
			)
		layers = []
		layers.append(block(self.in_channels, channels, conv_builder, stride, downsample))

		self.in_channels = channels * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_channels, channels, conv_builder))

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
		BASIC_STEM_BLOCK,
		**kwargs,
	)


def mc3_18(**kwargs: Any) -> VideoResNet:
	return _video_resnet(
		BasicBlock,
		[Conv3DSimple] + [Conv3DNoTemporal] * 3,  # type: ignore[list-item]
		[2, 2, 2, 2],
		BASIC_STEM_BLOCK,
		**kwargs,
	)


def r2plus1d_18(**kwargs: Any) -> VideoResNet:
	return _video_resnet(
		BasicBlock,
		[Conv2Plus1D] * 4,
		[2, 2, 2, 2],
		R2Plus1dStem,
		**kwargs,
	)


def profile(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
	net = r2plus1d_18()
	print(f"Num. params of R(2+1)D: {profile(r2plus1d_18())}")
	print(f"Num. params of R3D: {profile(r3d_18())}")
	print(f"Num. params of MC3: {profile(mc3_18())}")
