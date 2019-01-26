import os
import pickle
import typing

import numpy as np
import torch
from torch.utils import data

from config import Config
from logger import Logger


def putOnDevice ( t: torch.Tensor ):
	if Config.useCuda ():
		return t.cuda ()
	return t.cpu ()


def logTensorInfo ( t: torch.Tensor, name: str ):
	Logger.log ( "Data Type of {}: {}".format ( name, t.dtype ) )
	Logger.log ( "Device of {}: {}".format ( name, t.device ) )


def createValidationDataFile ( filenameTrain: str, filenameValidate: str, percentage: float = .20 ) -> None:
	"""
	Creates the validation file based off of the provided training file. Takes a percentage of the
	training file's lines and puts them in the validation file. If the validation file already exists,
	none of this happens.

	:param filenameTrain: The training file that stores all of the labels and paths of each image for training.
	:param filenameValidate: The validation file that stores all of the labels and paths of each image to be used for validation.
	:param percentage: The amount of images that are to be taken from the training file and put into the validation file.
	"""
	try:
		_ = open ( filenameValidate, 'r' )
		# If we try to read a non-existent file, we'll get an IOError.
		Logger.warn ( "Cannot create a new validation file since there's already one available" )
		return
	except IOError:
		Logger.log ( "Did not find validation file. That's good!" )

	Logger.log ( "Creating validation data file.", logger = "main" )
	trainingFile = open ( filenameTrain, "r" )
	if not trainingFile:
		Logger.error ( "Cannot find " + filenameTrain )
		return
	lines: typing.List[str] = trainingFile.readlines ()
	trainingFile.close ()
	# Header lines at the top of training data.
	headerLines: typing.List[str] = lines[:3]
	# The actual data that we want from training data.
	lines: typing.List[str] = lines[3:]

	# Now get rid of files that will be in the validation data
	trainingFile = open ( filenameTrain, "w" )
	# Write the header lines so that dataset.py will work correctly.
	for header in headerLines:
		trainingFile.write ( header )

	validationStepSize: int = int ( 100.0 * percentage )

	linesForValidationFile: typing.List[str] = lines[::validationStepSize]

	for line in lines:
		if line in linesForValidationFile:
			continue
		trainingFile.write ( line )
	trainingFile.close ()

	validationFile = open ( filenameValidate, "w+" )

	for header in headerLines:
		validationFile.write ( header )

	for line in linesForValidationFile:
		validationFile.write ( line )
	validationFile.close ()


def saveModel ( model_filename: str, schedulers: typing.List, epoch: int, network: torch.nn.Module,
                optimizer: torch.optim.Optimizer, validationLoss: typing.List[float], trainingLoss: typing.List[float],
                trainingDifference: typing.List[typing.Tuple[float, float]],
                validationDifference: typing.List[typing.Tuple[float, float]],
                uncertainty: typing.List[typing.Tuple[float, float]],
                testingDifference: typing.List[typing.Tuple[float, float]],
                anees: typing.List[typing.Tuple[float, float]] ) -> None:
	"""

	Saves the given model to the given file. Also saves the scheduler, last epoch, optimizer, and previous losses.

	:param model_filename: The file to save the model to.
	:param schedulers: The schedulers to save.
	:param epoch: The last epoch that the model was trained on.
	:param network: The network that is being saved.
	:param optimizer: The optimizer that is to be saved with the network.
	:param validationLoss: The history of validation losses.
	:param trainingLoss: The history of training losses.
	:param trainingDifference: The training differences over the epochs.
	:param validationDifference: The validation differences over the epochs.
	:param uncertainty: The uncertainty history of the network.
	:param testingDifference: The testing differences of the network over the epochs.
	:param anees: The history of Average Normalized Error Estimate Squared the network has had.
	"""

	# Don't save if we don't want to.
	if Config.getArgs ().dont_save:
		return

	# Save network to file.
	if "{}" in model_filename:
		model_filename = model_filename.format ( Config.version, epoch, Config.getArgs ().model_number )

	Logger.log ( "Saving model to " + model_filename + ".", logger = "min" )
	saveCheckpoint ( filepath = model_filename,
	                 currModel = { "model":                network.state_dict (),
	                               "epoch":                epoch,
	                               "optimizer":            optimizer.state_dict (),
	                               "schedulers":           schedulers,
	                               "version":              Config.version,
	                               "trainingLoss":         trainingLoss,
	                               "validationLoss":       validationLoss,
	                               "validationDifference": validationDifference,
	                               "trainingDifference":   trainingDifference,
	                               "uncertainty":          uncertainty,
	                               "testingDifference":    testingDifference,
	                               "anees":                anees
	                               } )


# This saves the model and the current settings (learning rate, optimizer, scheduler, etc)
# filepath is the path to the saved file.
# currModel is a dictionary that holds the model and the current settings
def saveCheckpoint ( filepath: str, currModel: typing.Dict[str, typing.Union[torch.nn.Module, torch.optim.Optimizer,
                                                                             typing.List, float, float, int,
                                                                             typing.Tuple[float, float],
                                                                             typing.Tuple[float, float], typing.Tuple[
	                                                                             float, float], typing.Tuple[
	                                                                             float, float], typing.Tuple[
	                                                                             float, float]]] ) -> None:
	"""
	Saves the given model to the given filepath, while also saving the given model to a file named
	posenet-latest-v<version number>.model

	:param filepath: The file to save the model at.
	:param currModel: The model to save.
	"""
	Logger.log ( "Saving network.", logger = "main" )
	torch.save ( f = filepath, obj = currModel )
	torch.save (
			f = "models/posenet-latest-v{}-N{:02d}.model".format ( Config.version, Config.getArgs ().model_number ),
			obj = currModel )


def printHeader () -> None:
	"""
	Print a header that says RECREATING POSENET. Used to seperate between different log files in the command line.

	"""
	print ( "{:^40}".format ( "=" * 40 ) )
	print ( "{:^40}".format ( "RECREATING POSENET" ) )
	print ( "{:^40}".format ( "v" + Config.version ) )
	print ( "{:^40}".format ( "=" * 40 ) )


def getMeanAndStd ( dataset: data.Dataset ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
	"""
	Computes the mean and standard deviation of the provided dataset.

	:param dataset: The dataset to be computed.
	:return: a tuple in the form of (mean, standard deviation)
	"""
	# Get some data
	filename = os.path.join ( Config.getArgs ().database_root, "norm.txt" )
	if os.path.isfile ( filename ):
		f = torch.load ( filename )
		return f[0], f[1]
	else:
		# Get some data
		dataloader = torch.utils.data.DataLoader ( dataset, batch_size = 1, shuffle = True,
		                                           num_workers = Config.getArgs ().threads )
		mean: torch.Tensor = torch.zeros ( 3 )
		std: torch.Tensor = torch.zeros ( 3 )
		Logger.log ( 'Computing mean and std.', logger = "main" )

		# for the data in the dataset
		for _, (data, labels) in enumerate ( dataloader ):
			# for each rgb value in the image
			for i in range ( 3 ):
				# Get the mean and standard deviation of the channel
				mean[i] += data[:, i, :, :].mean ()
				std[i] += data[:, i, :, :].std ()
		# Get the average mean and standard deviation
		mean.div_ ( len ( dataset ) )
		std.div_ ( len ( dataset ) )
		combined = torch.zeros ( 2, 3 )
		combined[0] = mean
		combined[1] = std
		torch.save ( combined, filename )
		return mean, std


def getPretrainedModel ( path: str ) -> typing.Dict[str, typing.Union[np.ndarray, torch.Tensor]]:
	"""
	Get the pretrained model from the given path. This should only be used if you are starting anew.

	:param path: The path that the pretrained model is at.
	:return: The pretrained model.
	"""
	Logger.log ( "Loading pretrained model from path of {}".format ( path ) )
	try:
		# Load the pretrained model at the given path
		data = torch.load ( f = path, map_location = Config.getDevice () )
		return data
	# If we have some unicode error (this happens when we try to load a PyTorch model from 0.3)
	except UnicodeDecodeError:
		with open ( path, 'rb' ) as f:
			# Load it with a different encoding that somehow works.
			data = pickle.load ( f, encoding = 'latin1' )
			return data
