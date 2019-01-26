import os
import typing

import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils import data

import util
from config import Config
from dataset import PoseNetDataSet
from logger import Logger
from posenet import PoseNet


def getErrorVector ( output: np.ndarray, target: np.ndarray ) -> typing.Tuple[np.ndarray, np.ndarray]:
	"""
	Subtracts the output from the target elementwise then averages out the differences per dimension (X Y Z W P Q R).

	:param output: The output of the network.
	:param target: The labels of the data.
	:return: The average of the elementwise subtraction of the output and the target with position being [0] and orientation being [1].
	"""

	err: np.ndarray = np.subtract ( output, target )
	# Take the average of all the x y z w p q r values within the columns.
	err = np.mean ( err, axis = 0 )

	return err[:3], err[3:]


def ANEES ( error: np.ndarray, uncertainty: np.ndarray, numOfImages: int ) -> float:
	"""
	ANEES (Average Normalized Error Estimate Squared). It's the mean of the error * the inverse of the uncertainty *
	the transpose of the error.

	:param error: The error of the network between the output and the target.
	:param uncertainty: The covariance matrix of the network's output.
	:param numOfImages: How many images are in the dataset.
	:return: The average of all of that.
	"""
	anees: np.ndarray = np.zeros ( numOfImages )
	for j in range ( numOfImages ):
		anees[j] = error[j] @ np.linalg.inv ( uncertainty[j] ) @ error[j].transpose ()
	return anees.mean ( dtype = float )


def getUncertainty ( output: np.ndarray, returnMatrix: bool = False ) -> typing.Union[
	typing.Tuple[float, float], typing.Tuple[np.ndarray, np.ndarray, float, float]]:
	"""
	Gets the uncertainty of the output from the network. If return matrix is true, the function will also return
	the covariance matrix along with the uncertainty values.

	:param output: the output from the network.
	:param returnMatrix: whether to return the covariance matrix or not.
	:return: the uncertainty values and the covariance matrix if returnMatrix is true.
	"""
	outPos: np.ndarray = output[:, :3]
	outOri: np.ndarray = output[:, 3:]

	covPos: np.ndarray = np.cov ( outPos, rowvar = 0 )
	covOri: np.ndarray = np.cov ( outOri, rowvar = 0 )

	uncertaintyPos: float = covPos.diagonal ().sum ()
	uncertaintyOri: float = covOri.diagonal ().sum ()
	if returnMatrix:
		return covPos, covOri, uncertaintyPos, uncertaintyOri
	return uncertaintyPos, uncertaintyOri


def getDifference ( output: torch.Tensor, target: torch.Tensor ) -> typing.Tuple[np.ndarray, np.ndarray]:
	"""
	Return the difference between the expected output of the network and the actual output of the network.
	Basically a wrapper for getErrorVector to do torch.Tensor's.

	:param output: Given output from the network
	:param target: Expected output of the network
	:return: a dictionary of the position and orientation differences between the two parameters.
	"""
	return getErrorVector ( output = output.cpu ().detach ().numpy (),
	                        target = target.cpu ().detach ().numpy () )


# Loss function for posenet
# outputPosition and outputOrientation are both the output provided by the neural network
# target is the output that is expected from the neural network
# beta is a constant to keep errors from data collection from getting in the way
# The target Tensor is supposed to be in the form of [batch_size, 7]
# with the second dimension being [x, y, z, w, p, q, r]
# where the position vector is [x, y, z]
# and the orientation vector is [w, p, q, r]
def posenetLoss ( output: torch.Tensor, target: torch.Tensor, beta: int ) -> torch.Tensor:
	"""
	The loss function defined in the original paper of PoseNet.

	:param output: The given output from the network
	:param target: The expected output from the network
	:param beta: The error constant
	:return: The loss of the network
	"""

	Logger.log ( "Computing loss." )
	Logger.log (
			"Types of things: output {} target {} beta {}".format ( type ( output ), type ( target ), type ( beta ) ) )
	Logger.log ( "Data types of things: output {} target {}".format ( output.dtype, target.dtype ) )
	Logger.log ( "Is on Cuda? Output {} target {}".format ( output.device, target.device ) )
	# Normalize the difference of the two position vectors
	positionLoss: torch.Tensor = torch.norm ( output[:, :3] - target[:, :3] )
	# Normalize the difference of the two orientation vectors
	# and multiply by the error constant.
	# Normalizing is the same thing as finding the unit vector.
	orientationDifference: torch.Tensor = torch.norm ( output[:, 3:] ) - torch.norm ( target[:, 3:] )
	orientationLoss: torch.Tensor = torch.norm ( orientationDifference ) * beta
	# return their sum
	return positionLoss + orientationLoss


def trainNetwork ( network: torch.nn.Module, optimizer: torch.optim.Optimizer, dataloader: data.DataLoader, epoch: int,
                   beta: int, print_every: int ) -> typing.Tuple[float, typing.Union[float, np.ndarray], np.ndarray]:
	"""
	Trains the network with the given parameters.

	:param network: The network that is to be trained.
	:param optimizer: The optimizer to teach the network what to do.
	:param dataloader: The data that the network will train on.
	:param epoch: The current epoch of this training session
	:param beta: The error constant to be used in the loss function
	:param print_every: Print loss and difference every x batches.
	:return: the average loss and differences of this training session.
	"""
	Logger.log ( "Training!", logger = "min" )
	# get the training data from the dataloader with the current batch number
	past_losses: typing.List[torch.Tensor] = []
	# Previous differences
	pastPosDiff: typing.List[np.ndarray] = []
	pastOriDiff: typing.List[np.ndarray] = []

	Logger.log ( "Epoch {}".format ( epoch ) )
	for batchNum, (data, labels) in enumerate ( dataloader ):
		# Log current batch number
		Logger.log ( "Training Batch #{}".format ( batchNum ) )

		labels: torch.Tensor = labels.float ()

		# Set network to start training.
		network.train ()

		Logger.log ( "Getting data" )
		# get the current batch.

		util.logTensorInfo ( data, "data" )
		util.logTensorInfo ( labels, "labels" )

		# If we're working with GPUs, put them on there.
		data = util.putOnDevice ( data )
		labels = util.putOnDevice ( labels )

		util.logTensorInfo ( data, "data" )
		util.logTensorInfo ( labels, "labels" )

		Logger.log ( "Resetting optimizer." )
		# reset the optimizer (right?)
		optimizer.zero_grad ()

		Logger.log ( "Getting output from network." )
		# run the network based on the current batch
		output: torch.Tensor = network ( data )

		util.logTensorInfo ( output, "output" )

		# Put the output onto the GPU
		output = util.putOnDevice ( output )

		util.logTensorInfo ( output, "output" )

		Logger.log ( "Calculating loss." )

		# calculate the error between the output and the expected output
		loss: torch.Tensor = posenetLoss ( output = output, target = labels, beta = beta )

		# This is a 1 dimensional vector that is the average of the differences between the output and target.
		difference: typing.Tuple[np.ndarray, np.ndarray] = getDifference ( output = output, target = labels )
		pastPosDiff.append ( difference[0] )
		pastOriDiff.append ( difference[1] )
		past_losses.append ( loss.detach ().cpu ().numpy () )

		# Put the loss onto the GPU if we can use it.
		if Config.useCuda ():
			loss = loss.cuda ()

		Logger.log ( "Calculating gradient." )
		# Calculate the gradient necessary to fix the network
		loss.backward ()

		Logger.log ( "Optimizing network" )
		# Assign those changes from the loss function to the network to
		# find the local (and preferably absolute) minimum
		optimizer.step ()

		# Check the accuracy every nth batches.
		if batchNum % print_every == 0:
			Logger.log ( "At epoch %i, minibatch %i. Loss: %.4f." % (epoch, batchNum, past_losses[-1]), logger = "min" )
			Logger.log ( "Error of position and orientation: {} {}".format (
					pastPosDiff[-1], np.linalg.norm ( pastOriDiff[-1] ) ), logger = "min" )
			Logger.log ( "Progress: {:.2%}".format ( (batchNum / len ( dataloader )) ), logger = "main" )

	avgLoss: float = np.average ( past_losses )
	# Print average losses
	Logger.log ( "Average training losses for epoch {}: {}".format ( epoch, avgLoss ), logger = "min" )
	avgPositionDifference: np.ndarray = np.average ( pastPosDiff, axis = 0 )
	avgOrientationDifference: np.ndarray = np.linalg.norm ( np.average ( pastOriDiff, axis = 0 ) )

	Logger.log ( "Average training difference for epoch {}: Pos: {} Ori: {}".format ( epoch, avgPositionDifference,
	                                                                                  avgOrientationDifference ),
	             logger = "min" )

	return avgLoss, np.linalg.norm ( avgPositionDifference ), avgOrientationDifference


def validate ( network: torch.nn.Module, dataloader: data.DataLoader, beta: int, print_every: int,
               epoch: int ) -> typing.Tuple[float, float, float]:
	"""
	Validates that the network trained effectively based on the given parameters.

	:param network: The network to be validated.
	:param dataloader: The data that will be used to validate
	:param beta: The data error constant that is to be used with the loss function
	:param print_every: For x batches, print the current loss and the differences of the network
	:param epoch: The current epoch
	:return: The average validation loss and difference of this validation session
	"""
	# Begin validation.
	Logger.log ( "Validating data!", logger = "min" )
	# Turn the network onto testing mode.
	network.eval ()

	# record past validation losses.
	validation_past_losses: typing.List[torch.Tensor] = []

	pastPosDiff: typing.List[np.ndarray] = []
	pastOriDiff: typing.List[np.ndarray] = []

	# Go through each thing in the testing data.
	for batchNum, (data, labels) in enumerate ( dataloader ):
		# Log current batch number
		Logger.log ( "Validation Batch #{}".format ( batchNum ) )

		labels: torch.Tensor = labels.float ()

		Logger.log ( "Getting data" )
		# If we want to use the GPU, use it.
		if Config.useCuda ():
			labels = labels.cuda ()
			data: torch.Tensor = data.cuda ()

		Logger.log ( "Getting output from network." )
		# Get the output of the network based on the given data.
		output: torch.Tensor = network ( data )
		# Put output onto the GPU.
		if Config.useCuda ():
			output = output.cuda ()

		Logger.log ( "Calculating loss" )

		# calculate the error between the output and the expected output
		loss: torch.Tensor = posenetLoss ( output = output, target = labels, beta = beta )
		differences: typing.Tuple[np.ndarray, np.ndarray] = getDifference ( output = output, target = labels )
		# These contain vectors.
		pastPosDiff.append ( differences[0] )
		pastOriDiff.append ( differences[1] )

		validation_past_losses.append ( loss.detach ().cpu ().numpy () )

		if batchNum % print_every == 0:
			Logger.log ( "Validation loss for epoch {} batch #{}: {}".format ( epoch, batchNum,
			                                                                   validation_past_losses[-1] ),
			             logger = "min" )
			Logger.log ( "Error of position and orientation: {} {}".format ( pastPosDiff[-1],
			                                                                 np.linalg.norm ( pastOriDiff[-1] ) ),
			             logger = "min" )
			Logger.log ( "Progress: {:.2%}".format ( (batchNum / len ( dataloader )) ), logger = "main" )
	avgLoss: float = np.average ( validation_past_losses )
	Logger.log ( "Total average validation loss: " + str ( avgLoss ), logger = "min" )

	# Compute average of position diff in the columns (per dimension like X Y Z)
	avgPositionDifference: float = np.average ( pastPosDiff, axis = 0 )
	# Get the average per column then get the magnitude of the average difference in orientation.
	avgOrientationDifference: float = np.linalg.norm ( np.average ( pastOriDiff, axis = 0 ) )

	Logger.log ( "Average validation difference for epoch {}: Pos: {} Ori: {}".format ( epoch, avgPositionDifference,
	                                                                                    avgOrientationDifference ),
	             logger = "min" )
	return avgLoss, np.linalg.norm ( avgPositionDifference ), avgOrientationDifference


def testSingleImage ( network: torch.nn.Module, image: Image, transforms ) -> typing.Tuple[
	np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	Logger.log ( "Testing single image!", logger = "main" )
	data: torch.Tensor = transforms ( image )

	data = util.putOnDevice(data)

	Logger.log ( "Resizing image." )
	data.resize_ ( 1, 3, 224, 224 )

	if Config.useCuda():
		network = network.cuda()
	else:
		network = network.cpu()

	Logger.log ( "Triggering dropout layers." )
	network.train ()
	outputs: typing.List[torch.Tensor] = []

	for i in range ( Config.getArgs ().num_of_tests ):
		outputs.append ( network ( data ) )

	outs: np.ndarray = np.zeros ( (len ( outputs ), 7) )
	for i in range ( len ( outputs ) ):
		outs[i] = outputs[i].cpu ().detach ().numpy ()

	Logger.log ( "Getting uncertainty." )
	certPosMat, certOriMat, _, _ = getUncertainty ( output = torch.from_numpy ( outs ), returnMatrix = True )

	Logger.log ( "Getting actual output.", logger = "main" )
	network.eval ()
	output: np.ndarray = network ( data ).detach ().cpu ().numpy ()
	return output[0], outs, certPosMat, certOriMat


def test ( network: torch.nn.Module, dataloader: torch.utils.data.DataLoader ) -> typing.Tuple[
	float, float, float, float, float, float]:
	"""
	Uses testSingleImage() to test the network. Goes through each image individually  and gets the uncertainty, the error,
	and ANEES.

	:param network: the network to be tested on.
	:param dataloader: The testing data
	:return: average uncertainty, average error, and ANEES. Each are tuples in the format of (position, orientation)
	"""

	Logger.log ( "Testing!", logger = "min" )
	uncertainties: typing.List[typing.Tuple[np.ndarray, np.ndarray]] = []
	errors: typing.List[typing.Tuple[np.ndarray, np.ndarray]] = []

	# Trigger the dropout layers.
	network.train ()

	print_every = Config.getArgs ().print_every

	if not Config.getArgs ().test_print_lots:
		print_every *= Config.getArgs ().batch_size

	# The batch size is 1
	for batchNum, (data, labels) in enumerate ( dataloader ):
		if Config.useCuda ():
			data = data.cuda ()
			labels = labels.cuda ()
		outputs: typing.List[np.ndarray] = []
		for i in range ( Config.getArgs ().num_of_tests ):
			# Get 1 output from the network.
			output: torch.Tensor = network ( data )
			outputs.append ( output.detach ().cpu ().numpy () )
		# Concat output into a single np.ndarray
		outputNP: np.ndarray = np.zeros ( (len ( outputs ), outputs[0].shape[1]) )
		for i in range ( len ( outputs ) ):
			outputNP[i] = outputs[i]
		error: typing.Tuple[np.ndarray, np.ndarray] = getErrorVector ( outputNP,
		                                                               target = labels.cpu ().detach ().numpy () )
		uncertainty: typing.Tuple[np.ndarray, np.ndarray, float, float] = getUncertainty ( outputNP,
		                                                                                   returnMatrix = True )

		errors.append ( error )
		# A list of the uncertainty matrices
		uncertainties.append ( uncertainty[:2] )
		if batchNum % print_every == 0:
			Logger.log ( "Last Error: Pos {} Ori {}".format ( errors[-1][0], np.linalg.norm ( errors[-1][1] ) ),
			             logger = "min" )
			Logger.log ( "Last uncertainty: Pos {} Ori {}".format ( np.sqrt ( uncertainty[0].diagonal () ),
			                                                        np.sqrt ( uncertainty[3] ) ), logger = "min" )
			Logger.log ( "Progress: {:4}%".format ( 100.0 * (float ( batchNum ) / float ( len ( dataloader ) )) ),
			             logger = "min" )

	errorsNP: np.ndarray = np.ndarray ( (len ( errors ), 7) )
	for i in range ( len ( errors ) ):
		errorsNP[i] = np.concatenate ( errors[i], axis = 0 )
	posErrors: np.ndarray = errorsNP[:, :3]
	oriErrors: np.ndarray = errorsNP[:, 3:]

	averagePosErr: float = np.average ( posErrors, axis = 0 )
	averageOriErr: float = np.linalg.norm ( np.average ( oriErrors, axis = 0 ) )

	averageError: typing.Tuple[float, float] = (np.linalg.norm ( averagePosErr ), averageOriErr)

	Logger.log (
			"Average Error during test: Pos: {} Ori: {}".format ( averagePosErr, averageError[1] ),
			logger = "min" )

	certaintyNP: typing.Tuple[np.ndarray, np.ndarray] = (
		np.ndarray ( (len ( uncertainties ), 3, 3) ),
		np.ndarray ( (len ( uncertainties ), 4, 4) )
	)
	for i in range ( len ( uncertainties ) ):
		certaintyNP[0][i] = uncertainties[i][0]
		certaintyNP[1][i] = uncertainties[i][1]
	posCerts: np.ndarray = certaintyNP[0]
	oriCerts: np.ndarray = certaintyNP[1]

	avgPosCert: np.ndarray = np.ndarray ( (len ( posCerts ), 3) )
	for i in range ( len ( posCerts ) ):
		avgPosCert[i] = posCerts[i].diagonal ()
	avgPosCert = np.average ( avgPosCert, axis = 0 )

	avgOriCert: np.ndarray = np.ndarray ( (len ( oriCerts ), 4) )
	for i in range ( len ( oriCerts ) ):
		avgOriCert[i] = oriCerts[i].diagonal ()
	avgOriCert = np.average ( avgOriCert, axis = 0 )
	avgOriCert = avgOriCert.sum ()

	averageUncertainty: typing.Tuple[float, float] = (np.sqrt ( avgPosCert.sum () ), np.sqrt ( avgOriCert ))

	Logger.log ( "Average uncertainty: Pos {} Ori {}".format ( np.sqrt ( avgPosCert ),
	                                                           averageUncertainty[1] ) )

	aneesPos: float = ANEES ( posErrors, posCerts, len ( dataloader ) )
	aneesOri: float = ANEES ( oriErrors, oriCerts, len ( dataloader ) )

	Logger.log ( "Average Normalized Error Estimate Squared: Pos {} Ori {}".format ( aneesPos, aneesOri ),
	             logger = "min" )

	return averageUncertainty[0], averageUncertainty[1], averageError[0], averageError[1], aneesPos, aneesOri


def loadModel () -> typing.Dict[str, typing.Union[torch.nn.Module, torch.optim.Optimizer,
                                                  typing.List, float, float, int, typing.Tuple[float, float],
                                                  typing.Tuple[float, float], typing.Tuple[float, float], typing.Tuple[
	                                                  float, float], typing.Tuple[float, float]]]:
	"""
	Loads the model from file. If any keys mismatch, it'll wrap the network in a nn.DataParallel. If it gets any other errors
	it will initiate the network from the default GoogLeNet model found in pretrained-models. Also loads epochs, training
	and validation differences, and a plethora of other stuff. See util.saveModel for more information on what is loaded.

	:return: Everything that was in the file.
	"""
	# Create the network.
	network: torch.nn.Module = PoseNet ( input_nc = 3 )

	# Default network should be placed on the GPU so that the other default things
	# can expect that to be the case.
	if Config.useCuda ():
		network = network.cuda ()

	Logger.log ( "Creating learning objects.", logger = "main" )

	optimizer: torch.optim.Optimizer = torch.optim.Adam ( network.parameters (), lr = Config.getArgs ().learning_rate )

	scheduler1: torch.optim.lr_scheduler.ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau (
			optimizer = optimizer, threshold = Config.getArgs ().beta,
			verbose = Logger.shouldLog (), threshold_mode = "abs",
			factor = Config.getArgs ().factor )

	scheduler2: torch.optim.lr_scheduler.StepLR = torch.optim.lr_scheduler.StepLR ( optimizer = optimizer,
	                                                                                step_size = 80,
	                                                                                gamma = Config.getArgs ().factor )
	trainingLoss: typing.List[float] = []
	validationLoss: typing.List[float] = []
	trainingDiff: typing.List[float] = []
	validationDiff: typing.List[float] = []
	testingDiff: typing.List[float] = []
	uncertainty: typing.List[float] = []
	anees: typing.List[float] = []

	starting_epoch: int = 0

	hasParallelilized: bool = False

	defaultPretrainedModel: str = "pretrained-models/places-googlenet.pickle"

	Logger.log ( "Loading from pretrained model {}".format ( Config.getArgs ().pretrained_model ), logger = "main" )
	# Load from pretrained model.
	path: str = Config.getArgs ().pretrained_model
	if path is not None or Config.getArgs ().resume:
		try:
			# Load checkpoint from file
			checkpoint: typing.Dict[str, typing.Any] = util.getPretrainedModel ( path = path )
			# If we have version in the data of checkpoint
			if "version" in checkpoint:
				# But it's not the current version...
				if checkpoint["version"] != Config.version:
					# Load in GoogLeNet
					Logger.warn ( "User wants to load outdated model file!" )
				# If the version matches, load in the posenet model

				try:
					# We can't load in from googlenet here because googlenet doesn't have a version key
					network.load_state_dict ( checkpoint["model"] )
				except RuntimeError:
					# If we get a RuntimeError it's probably because we're trying to load something
					# that was wrapped in the nn.DataParallel layer.
					# We could prewrap it all beforehand, but that would mean we would need to
					# circumnavigate GoogLeNet and its keys.
					network = torch.nn.DataParallel ( network )
					hasParallelilized = True
					try:
						network.load_state_dict ( checkpoint["model"] )
					except RuntimeError:
						# if we still get an error message, that means we changed the architecture.
						Logger.warn (
								"Trying to load from version that has different architecture than current version." )
						Logger.log ( "Loading from default model.", logger = "min" )
						network = torch.nn.DataParallel (
								PoseNet ( input_nc = 3, weights = util.getPretrainedModel ( defaultPretrainedModel ) ) )
				if checkpoint["version"] == Config.version:
					# We wouldn't want to load in these things from a different version.
					optimizer.load_state_dict ( checkpoint["optimizer"] )
					scheduler1 = checkpoint["schedulers"][0]
					scheduler2.load_state_dict ( checkpoint["schedulers"][1] )
					trainingLoss = checkpoint["trainingLoss"]
					validationLoss = checkpoint["validationLoss"]
					trainingDiff = checkpoint["trainingDifference"]
					validationDiff = checkpoint["validationDifference"]
					starting_epoch = checkpoint["epoch"] + 1
					testingDiff = checkpoint["testingDifference"]
					uncertainty = checkpoint["uncertainty"]
					anees = checkpoint["anees"]
			else:  # If we don't have a version in the model file, we'll load GoogLeNet.
				Logger.log ( "Loading network from default pretrained model." )
				network = PoseNet ( input_nc = 3, weights = util.getPretrainedModel ( defaultPretrainedModel ) )
		except FileNotFoundError:
			Logger.error ( "Cannot find pretrained model file!" )
			Logger.log ( "Loading from default pretrained model." )
			# Load from googleNet
			network = PoseNet ( input_nc = 3, weights = util.getPretrainedModel ( defaultPretrainedModel ) )
	else:
		Logger.log ( "Loading from default pretrained model." )
		# Load from googleNet
		network = PoseNet ( input_nc = 3, weights = util.getPretrainedModel ( defaultPretrainedModel ) )

	# This will be the same on GPU or CPU so don't you worry baby.
	# Don't you worry OHHHH-OHOH
	if not hasParallelilized:
		network = torch.nn.DataParallel ( network )

	# If we have access to GPUs, put PoseNet on them.
	if Config.useCuda ():
		network = network.cuda ()

	return {
		"network":              network,
		"optimizer":            optimizer,
		"schedulers":           [scheduler1, scheduler2],
		"trainingLoss":         trainingLoss,
		"validationLoss":       validationLoss,
		"startingEpoch":        starting_epoch,
		"validationDifference": validationDiff,
		"trainingDifference":   trainingDiff,
		"testingDifference":    testingDiff,
		"uncertainty":          uncertainty,
		"anees":                anees
	}


def main ():
	"""
	The main training script of the program. Runs through the whole shebang!

	"""
	args = Config.getArgs ()
	if args.threads is 7:
		Logger.warn ( "Number of threads has been limited to 7 for PyTorch reasons." )
	if args.force_threads and args.threads >= 8:
		Logger.warn ( "Remember PyTorch can be a little wonky with 8+ threads!" )
	Logger.log ( "Use Cuda? {}".format ( Config.useCuda () ) )
	if Config.useCuda ():
		Logger.log ( "CUDA Properties is: {}".format ( torch.cuda.get_device_properties ( 0 ) ) )

	Logger.log ( "Parsed args", logger = "main" )
	Logger.log ( str ( args ) )

	# Create validation data by stealing some from training data.
	util.createValidationDataFile ( filenameTrain = os.path.join ( args.database_root, "dataset_train.txt" ),
	                                filenameValidate = os.path.join ( args.database_root, "dataset_validate.txt" ) )

	Logger.log ( "Creating PoseNet.", logger = "main" )

	# Get the mean and std of the database that we'll be using.
	# Btw this takes a while.
	Logger.log ( message = "Getting mean and std of dataset", logger = "main" )
	Logger.log ( message = "This can take a while...", logger = "main" )
	if not args.skip_training:
		normalizeDataset: torch.utils.data.Dataset = PoseNetDataSet ( dataroot = args.database_root,
		                                                              transform = tv.transforms.ToTensor () )
	else:
		normalizeDataset: torch.utils.data.Dataset = PoseNetDataSet ( dataroot = args.database_root,
		                                                              transform = tv.transforms.ToTensor (),
		                                                              phase = "test" )
	mean, std = util.getMeanAndStd ( normalizeDataset )

	Logger.log ( "Mean and std is {} {}".format ( mean, std ) )

	# Create the transformation functions we'll use on each of our images.
	transformImg: tv.transforms.Compose = tv.transforms.Compose ( [  # Resize the image to 256x256
		tv.transforms.Resize ( (256, 256) ),  # Crop the image to 224 x 224
		tv.transforms.RandomCrop ( (224, 224) ),  # Make the image a tensor with dimensions 3x256x256
		# (3 because of rgb values within the image)
		tv.transforms.ToTensor (),  # Normalize the images
		tv.transforms.Normalize ( mean = mean, std = std )] )

	Logger.log ( "Loading dataset.", logger = "main" )
	# Create the dataset based on the KingsCollege database that should be in
	# the working directory of the program.
	datasetTrain: PoseNetDataSet = PoseNetDataSet ( dataroot = args.database_root, transform = transformImg )
	datasetValid: PoseNetDataSet = PoseNetDataSet ( dataroot = args.database_root, transform = transformImg,
	                                                phase = "validate" )
	datasetTest: PoseNetDataSet = PoseNetDataSet ( dataroot = args.database_root, transform = transformImg,
	                                               phase = "test" )

	# Create the dataloader which takes in the KingsCollege dataset and
	# allows us to use it in our training.
	dataloaderTrain: data.DataLoader = data.DataLoader ( dataset = datasetTrain, batch_size = args.batch_size,
	                                                     shuffle = True,
	                                                     num_workers = Config.getArgs ().threads )
	dataloaderValidate: data.DataLoader = data.DataLoader ( dataset = datasetValid, batch_size = args.batch_size,
	                                                        shuffle = True,
	                                                        num_workers = Config.getArgs ().threads )
	# Same as above but with testing.
	dataloaderTest: data.DataLoader = data.DataLoader ( dataset = datasetTest, batch_size = 1,
	                                                    num_workers = Config.getArgs ().threads )

	# The typing for loadModel is already horrible. We won't do it here.
	bigDictionary = loadModel ()
	network = bigDictionary["network"]
	optimizer = bigDictionary["optimizer"]

	schedulers: typing.List = bigDictionary["schedulers"]
	schedulerPlateau: torch.optim.lr_scheduler.ReduceLROnPlateau = schedulers[0]
	schedulerStep: torch.optim.lr_scheduler.StepLR = schedulers[1]

	trainingLoss: typing.List[float] = bigDictionary["trainingLoss"]
	validationLoss: typing.List[float] = bigDictionary["validationLoss"]
	trainingDiff: typing.List[typing.Tuple[float, float]] = bigDictionary["trainingDifference"]
	validationDiff: typing.List[typing.Tuple[float, float]] = bigDictionary["validationDifference"]
	starting_epoch: int = bigDictionary["startingEpoch"]
	testUncertainties: typing.List[typing.Tuple[float, float]] = bigDictionary["uncertainty"]
	testDifferences: typing.List[typing.Tuple[float, float]] = bigDictionary["testingDifference"]
	anees: typing.List[typing.Tuple[float, float]] = bigDictionary["anees"]

	ending_epoch: int = starting_epoch + args.epochs

	Logger.log ( "Optimizer state: {}".format ( optimizer.state ) )

	if Config.useCuda ():
		network = network.cuda ()
	else:
		network = network.cpu ()

	# for each epoch that we want to do...
	for epoch in range ( starting_epoch, ending_epoch ):
		if not args.skip_training:
			# If the current epoch is divisible by 80, decrease it by 90%
			schedulerStep.step ()
			# train network
			trainLoss, trainPosDiff, trainOriDiff = trainNetwork ( network = network, optimizer = optimizer,
			                                                       dataloader = dataloaderTrain,
			                                                       epoch = epoch,
			                                                       beta = args.beta,
			                                                       print_every = args.print_every )
			trainingLoss.append ( trainLoss )
			trainingDiff.append ( (trainPosDiff, trainOriDiff) )

			# validate network
			validLoss, validPosDiff, validOriDiff = validate ( network = network, dataloader = dataloaderValidate,
			                                                   beta = args.beta,
			                                                   print_every = args.print_every, epoch = epoch )

			validationLoss.append ( validLoss )
			validationDiff.append ( (validPosDiff, validOriDiff) )

			# Decrease threshold of scheduler when our loss is less than the current threshold
			# times a certain factor.
			if validationLoss[-1] < schedulerPlateau.threshold * args.threshold_factor:
				schedulerPlateau.threshold /= 2

			schedulerPlateau.step ( epoch = epoch, metrics = validationLoss[-1] )

			# Save the network
			util.saveModel ( model_filename = args.model_file,
			                 schedulers = [schedulerPlateau, schedulerStep.state_dict ()],
			                 epoch = epoch, network = network, optimizer = optimizer,
			                 trainingLoss = trainingLoss, validationLoss = validationLoss,
			                 trainingDifference = trainingDiff, validationDifference = validationDiff,
			                 testingDifference = testDifferences, uncertainty = testUncertainties, anees = anees )

		if not args.skip_testing and (epoch % args.test_every == 0 or epoch + 1 == ending_epoch):
			# Test the network
			posUncert, oriUncert, posDiff, oriDiff, aneesPos, aneesOri = test ( network = network,
			                                                                    dataloader = dataloaderTest )
			testUncertainties.append ( (posUncert, oriUncert) )
			testDifferences.append ( (posDiff, oriDiff) )
			anees.append ( (aneesPos, aneesOri) )
			# Save the network
			util.saveModel ( model_filename = args.model_file,
			                 schedulers = [schedulerPlateau, schedulerStep.state_dict ()],
			                 epoch = epoch, network = network, optimizer = optimizer,
			                 trainingLoss = trainingLoss, validationLoss = validationLoss,
			                 trainingDifference = trainingDiff, validationDifference = validationDiff,
			                 testingDifference = testDifferences, uncertainty = testUncertainties, anees = anees )
