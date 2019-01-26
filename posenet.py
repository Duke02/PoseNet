import typing

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from logger import Logger
from util import logTensorInfo, putOnDevice


def weightsInitFromGoogleNet ( key: str, module: nn.Module,
                               weights: typing.Dict[str, typing.Union[np.ndarray, torch.Tensor]] = None ) -> nn.Module:
	"""
	Initialize weights for the given model based on the provided weights file.

	:param key: The layer's key that is to be initialized.
	:param module: The module whose weights are to be initialized.
	:param weights: The pretrained model in the form of a dictionary.
	:return: The initialized module.
	"""
	# If the key is not in weights
	if weights is not None and str ( key + "_1" ) not in weights.keys ():
		# Print an error message.
		Logger.error ( "Cannot find key: {}!".format ( key ) )
	# If we aren't using GoogLeNet or the key isn't in weights...
	if weights is None or str ( key + "_1" ) not in weights.keys ():
		# Initialize everything per PoseNet standards.
		init.constant_ ( module.bias, 0.0 )
		# If it's the position outputs
		if key == "XYZ":
			# Initialize them with a standard deviation of 0.5
			init.normal_ ( module.weight, 0, 0.5 )
		else:
			init.normal_ ( module.weight, 0, 0.01 )
	else:  # If we are using GoogLeNet and we can find the key...
		# Load in the module's data from GoogLeNet
		Logger.log ( "Loading from GoogLeNet with key {}".format ( str ( key ) ) )
		module.bias.data[...] = torch.from_numpy ( weights[str ( key + "_1" )] )
		module.weight.data[...] = torch.from_numpy ( weights[str ( key + "_0" )] )
	# Return our initialized module.
	return module


class MiddleOutput ( nn.Module ):
	"""
		The class for the outputs used by the network. Depending on the lossID, the output will be placed somewhere in the network, potentially at a discounted rate.

		:param lossID: which output this is. 1 and 2 are in the middle. 3 is the final one.
		:param weights: The pretrained model that is to be used if possible.
	"""

	def __init__ ( self, lossID: str, weights: typing.Dict[str, typing.Union[np.ndarray, torch.Tensor]] = None ):

		super ( MiddleOutput, self ).__init__ ()
		# If it's not the final output, load as necessary
		if lossID != "loss3":
			nc: typing.Dict[str, int] = { "loss1": 512, "loss2": 528 }
			self.projection = nn.Sequential (
					nn.AvgPool2d ( kernel_size = 5, stride = 3 ),
					weightsInitFromGoogleNet ( lossID + "/conv", nn.Conv2d ( nc[lossID], 128, kernel_size = 1 ),
					                           weights ),
					nn.ReLU ( inplace = False ) )
			self.cls_fc_pose = nn.Sequential (
					weightsInitFromGoogleNet ( lossID + "/fc", nn.Linear ( 2048, 1024 ), weights ),
					nn.ReLU ( inplace = False ),
					nn.Dropout ( 0.7 )
			)
			self.cls_fc_xy = weightsInitFromGoogleNet ( "XYZ", nn.Linear ( 1024, 3 ) )
			self.cls_fc_wpqr = weightsInitFromGoogleNet ( "WPQR", nn.Linear ( 1024, 4 ) )
		else:  # if it is the final output, load as necessary
			self.projection = nn.AvgPool2d ( kernel_size = 7, stride = 1 )
			self.cls_fc_pose = nn.Sequential (
					weightsInitFromGoogleNet ( "pose", nn.Linear ( 1024, 2048 ) ),
					nn.ReLU ( inplace = False ),
					nn.Dropout ( 0.5 )
			)
			self.cls_fc_xy = weightsInitFromGoogleNet ( "XYZ", nn.Linear ( 2048, 3 ) )
			self.cls_fc_wpqr = weightsInitFromGoogleNet ( "WPQR", nn.Linear ( 2048, 4 ) )

	def forward ( self, x: torch.Tensor ) -> torch.Tensor:
		"""
		Goes through the output with the given input.

		:param x: The given input.
		:return: The output of the network at this stage.
		"""
		xN: torch.Tensor = self.projection ( x )
		# The input's size at 0 is guaranteed to be the batch size,
		# while the second dimension depends on the output.
		xN = xN.view ( xN.size ( 0 ), -1 )
		xN = self.cls_fc_pose ( xN )
		out_xyz = self.cls_fc_xy ( xN )
		out_wpqr = self.cls_fc_wpqr ( xN )
		out_wpqr = F.normalize ( out_wpqr )
		return torch.cat ( (out_xyz, out_wpqr), 1 )


class InceptionBlock ( nn.Module ):
	"""
	The Inception Module that was created in the GoogLeNet paper. This is version 1.

	:param incp: ID of inception module.
	:param input_nc: Number of input channels
	:param x1_nc: Number of channels of x1
	:param x3_reduce_nc: Number of channels in x3_reduce
	:param x3_nc: Number of channels in x3
	:param x5_reduce_nc: Number of channels in x5_reduce
	:param x5_nc: Number of channels in x5
	:param proj_nc: Number of channels in the projection branch
	:param weights: Weights that might be from pretrained model
	"""

	def __init__ ( self, incp: str, input_nc: int, x1_nc: int, x3_reduce_nc: int, x3_nc: int, x5_reduce_nc: int,
	               x5_nc: int, proj_nc: int, weights: typing.Dict[str, typing.Union[np.ndarray, torch.Tensor]] = None ):
		super ( InceptionBlock, self ).__init__ ()

		self.branch_x1 = nn.Sequential (
				weightsInitFromGoogleNet ( "inception_" + incp + "/1x1", nn.Conv2d ( input_nc, x1_nc, kernel_size = 1 ),
				                           weights ),
				nn.ReLU ( inplace = False ) )
		self.branc_x3 = nn.Sequential (
				weightsInitFromGoogleNet ( "inception_" + incp + "/3x3_reduce",
				                           nn.Conv2d ( input_nc, x3_reduce_nc, kernel_size = 1 ), weights ),
				nn.ReLU ( inplace = False ),
				weightsInitFromGoogleNet ( "inception_" + incp + "/3x3",
				                           nn.Conv2d ( x3_reduce_nc, x3_nc, kernel_size = 3, padding = 1 ), weights ),
				nn.ReLU ( inplace = False )
		)
		self.branch_x5 = nn.Sequential (
				weightsInitFromGoogleNet ( "inception_" + incp + "/5x5_reduce",
				                           nn.Conv2d ( input_nc, x5_reduce_nc, kernel_size = 1 ), weights ),
				nn.ReLU ( inplace = False ),
				weightsInitFromGoogleNet ( "inception_" + incp + "/5x5",
				                           nn.Conv2d ( x5_reduce_nc, x5_nc, kernel_size = 5, padding = 2 ), weights ),
				nn.ReLU ( inplace = False )
		)
		self.branc_proj = nn.Sequential (
				nn.MaxPool2d ( kernel_size = 3, stride = 1, padding = 1 ),
				weightsInitFromGoogleNet ( "inception_" + incp + "/pool_proj",
				                           nn.Conv2d ( input_nc, proj_nc, kernel_size = 1 ), weights ),
				nn.ReLU ( inplace = False )
		)

		if incp in ["3b", "4e"]:
			self.pool = nn.MaxPool2d ( kernel_size = 3, stride = 2, padding = 1 )
		else:
			self.pool = None

	def forward ( self, x: torch.Tensor ) -> torch.Tensor:
		"""
		Goes through the inception module with the given input.

		:param x: The given output.
		:return: A concatenation of each of the branches outputs.
		"""
		outputs: typing.List[torch.Tensor] = [self.branch_x1 ( x ), self.branc_x3 ( x ), self.branch_x5 ( x ),
		                                      self.branc_proj ( x )]
		xN: torch.Tensor = torch.cat ( outputs, 1 )
		if self.pool is not None:
			return self.pool ( xN )
		return xN


class PoseNet ( nn.Module ):
	"""
	The main network of this project. See the README.md for the link to the document.

	:param input_nc: The number of input channels. 3 if rgb, 1 if grayscale images are passed in.
	:param weights: The pretrained model to base our initial weights off of.
	:param isTraining: Is the network training? Or is it testing?
	"""

	def __init__ ( self, input_nc: int, weights: typing.Dict[str, typing.Union[torch.Tensor, np.ndarray]] = None,
	               isTraining: bool = True ):
		super ( PoseNet, self ).__init__ ()

		if weights is not None:
			Logger.log ( "Loading from GoogLeNet." )

		self.isTraining: bool = isTraining

		self.before_inception = nn.Sequential (
				weightsInitFromGoogleNet ( "conv1/7x7_s2",
				                           nn.Conv2d ( input_nc, 64, kernel_size = 7, stride = 2, padding = 3 ),
				                           weights ),
				nn.ReLU ( inplace = False ),
				nn.MaxPool2d ( kernel_size = 3, stride = 2, padding = 1 ),
				nn.LocalResponseNorm ( size = 5 ),
				weightsInitFromGoogleNet ( "conv2/3x3_reduce", nn.Conv2d ( 64, 64, kernel_size = 1 ) ),
				nn.ReLU ( inplace = False ),
				weightsInitFromGoogleNet ( "conv2/3x3", nn.Conv2d ( 64, 192, kernel_size = 3, padding = 1 ), weights ),
				nn.ReLU ( inplace = False ),
				nn.LocalResponseNorm ( size = 5 ),
				nn.MaxPool2d ( kernel_size = 3, stride = 2, padding = 1 )
		)

		self.inception_3a = InceptionBlock ( "3a", 192, 64, 96, 128, 16, 32, 32, weights )
		self.inception_3b = InceptionBlock ( "3b", 256, 128, 128, 192, 32, 96, 64, weights )
		self.inception_4a = InceptionBlock ( "4a", 480, 192, 96, 208, 16, 48, 64, weights )
		self.inception_4b = InceptionBlock ( "4b", 512, 160, 112, 224, 24, 64, 64, weights )
		self.inception_4c = InceptionBlock ( "4c", 512, 128, 128, 256, 24, 64, 64, weights )
		self.inception_4d = InceptionBlock ( "4d", 512, 112, 144, 288, 32, 64, 64, weights )
		self.inception_4e = InceptionBlock ( "4e", 528, 256, 160, 320, 32, 128, 128, weights )
		self.inception_5a = InceptionBlock ( "5a", 832, 256, 160, 320, 32, 128, 128, weights )
		self.inception_5b = InceptionBlock ( "5b", 832, 384, 192, 384, 48, 128, 128, weights )

		self.dp1 = nn.Sequential (
				nn.Dropout ( 0.5 )
		)

		self.cls1_fc = MiddleOutput ( lossID = "loss1", weights = weights )
		self.cls2_fc = MiddleOutput ( lossID = "loss2", weights = weights )
		self.cls3_fc = MiddleOutput ( lossID = "loss3", weights = weights )

	def forward ( self, x: torch.Tensor ) -> torch.Tensor:
		"""
		Goes through the network with the given input and outputs a dictionary based off of whether the
		network is training or not.

		:param x: the given input.
		:return: A dictionary with keys position and orientation. If the network is training, it will also output the auxillary outputs if necessary.
		"""
		Logger.log ( "Running through network!" )
		x = putOnDevice ( x )
		# We do a copy of the input just to make sure that we don't write to the input.
		# This is because nn.DataParallel can apparently get finicky with writing to the input
		# during a model's forward pass.
		xN: torch.Tensor = self.before_inception ( x )
		xN = self.inception_3a ( xN )
		xN = self.inception_3b ( xN )
		xN = self.inception_4a ( xN )
		if self.isTraining:
			mo1: torch.Tensor = self.cls1_fc ( xN )
		xN = self.inception_4b ( xN )
		xN = self.inception_4c ( xN )
		xN = self.inception_4d ( xN )
		if self.isTraining:
			mo2: torch.Tensor = self.cls2_fc ( xN )
		xN = self.inception_4e ( xN )
		xN = self.inception_5a ( xN )
		xN = self.inception_5b ( xN )
		xN = self.dp1 ( xN )
		out: torch.Tensor = self.cls3_fc ( xN )
		if self.isTraining:
			out += mo1 * 0.3 + mo2 * 0.3
		return out
