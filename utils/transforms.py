from torch import nn, Tensor


class ConvertBCHWtoCBHW(nn.Module):
	"""Convert tensor from (B, C, H, W) to (C, B, H, W)"""

	def forward(self, vid: Tensor) -> Tensor:
		return vid.permute(1, 0, 2, 3)


class ConvertTHWCtoTCHW(nn.Module):
	"""Convert tensor from (T, H, W, C) to (T, C, H, W)"""

	def forward(self, vid: Tensor) -> Tensor:
		return vid.permute(0, 3, 1, 2)
