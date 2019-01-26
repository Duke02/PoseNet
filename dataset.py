import os
import typing

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

import util


class PoseNetDataSet ( data.Dataset ):
	"""
	The dataset of posenet. Uses Kings College database by default.

	Loads the labels and images with the
	assumption that they are in a file called 'dataset_<phase>.txt'. Each line within the dataset
	text file must use the convention of 'path/to/image X Y Z W P Q R'. Please organize your data appropriately.

	:param transform: The transformations of the dataset. Must contain torchvision.transforms.ToTensor()
	:param phase: The phase of the dataset that will be used to find the data file.
	:param dataroot: The root where the data is.

	"""

	def __init__ ( self, transform = None, phase: str = "train", dataroot: str = "KingsCollege",
	               singleImage: typing.Union[None, str] = None ):
		super ( PoseNetDataSet, self ).__init__ ()
		self.root = dataroot
		self.phase = phase

		# Load all of the images and their provided poses from the dataset file within KingsCollege.
		labelsFile = os.path.join ( self.root, "dataset_" + self.phase + ".txt" )
		self.imagePaths = np.loadtxt ( labelsFile, dtype = str, delimiter = ' ', skiprows = 3, usecols = 0 )
		self.imagePaths = [os.path.join ( self.root, path ) for path in self.imagePaths]
		self.poses: typing.List[float] = np.loadtxt ( labelsFile, dtype = float, delimiter = ' ', skiprows = 3,
		                                              usecols = (1, 2, 3, 4, 5, 6, 7) )
		if singleImage is not None:
			indexOfThing = 0
			for i in range ( len ( self.imagePaths ) ):
				if singleImage in self.imagePaths[i]:
					indexOfThing = i
					break
			self.imagePaths = [self.imagePaths[indexOfThing]]
			self.poses = [self.poses[indexOfThing]]

		self.transform = transform

	def __getitem__ ( self, key: typing.Union[int, str] ) -> typing.Union[typing.Tuple[
		                                                                      typing.Union[
			                                                                      Image.Image, torch.Tensor], float], float, None]:
		"""
		Gets the image at the provided image. To be used alongside DataLoader.

		:param key: The index of the image within the dataset.
		:return: A tuple in the form of (image, label)
		"""
		if type ( key ) is int:
			# Get the index
			indexOfImg = key % len ( self )
			# Get the path of the needed image
			imgPath = self.imagePaths[indexOfImg]
			# Get the image from the path
			img = Image.open ( imgPath ).convert ( 'RGB' )
			# Get the label of the image from the index
			pose: float = self.poses[indexOfImg]

			# Transform the image if we have some.
			if self.transform is not None:
				img = self.transform ( img )

			if type ( img ) == torch.Tensor:
				img = util.putOnDevice ( img )
			# Return a tuple in the form of (data, label)
			return img, pose
		else:
			for i in range ( len ( self.imagePaths ) ):
				if self.imagePaths[i] == key:
					return self.poses[i]
			return None

	def __len__ ( self ) -> int:
		"""

		:return: the number of image paths the dataset contains.
		"""
		return len ( self.imagePaths )
