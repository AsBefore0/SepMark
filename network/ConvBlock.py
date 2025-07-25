import torch.nn as nn


class ConvINRelu(nn.Module):
	"""
	A sequence of Convolution, Instance Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride):
		super(ConvINRelu, self).__init__()

		self.layers = nn.Sequential(
			# 3x3 卷积
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			# InstanceNorm
			nn.InstanceNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ConvBlock(nn.Module):
	'''
	Network that composed by layers of ConvINRelu
	'''

	def __init__(self, in_channels, out_channels, blocks=1, stride=1):
		super(ConvBlock, self).__init__()

		# 只有第一层的步长自己设置  方便复用
		layers = [ConvINRelu(in_channels, out_channels, stride)] if blocks != 0 else []
		# 后续的步长固定为 1
		for _ in range(blocks - 1):
			# (B, C, H, W) -> (B, C, H, W)
			layer = ConvINRelu(out_channels, out_channels, 1)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
