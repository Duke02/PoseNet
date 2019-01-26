import typing

import matplotlib
import torch
import torchvision as tv
from PIL import Image
from matplotlib.axes import Axes

import dataset
from logger import Logger

matplotlib.use ( "Agg" )
import matplotlib.pyplot as plt
import numpy as np

import train
from config import Config


def getBounds ( axis: Axes ):
	bottom, top = axis.get_ylim ()
	left, right = axis.get_xlim ()
	height = 0.5 * (abs ( bottom ) + abs ( top ))
	width = 0.5 * (abs ( right ) + abs ( left ))
	return bottom, top, left, right, height, width


def plotTrendLine ( axis: Axes, x_axis: typing.Union[np.ndarray, typing.List],
                    y_axis: typing.Union[np.ndarray, typing.List], color, label: str = None ) -> None:
	z: np.ndarray = np.polyfit ( x_axis, y_axis, 1 )
	p = np.poly1d ( z )
	axis.plot ( x_axis, p ( x_axis ), label = label, color = color, alpha = 0.75 )


def plotLosses ():
	bigDictionary: typing.Dict = train.loadModel ()

	epoch: int = bigDictionary["startingEpoch"]
	step: int = min ( epoch // 25 + 1, 10 )

	trainingLoss: typing.List[float] = bigDictionary["trainingLoss"][::step]
	validationLoss: typing.List[float] = bigDictionary["validationLoss"][::step]

	beta: int = Config.getArgs ().beta

	epochs: np.ndarray = np.arange ( 0, epoch, step )
	betas: np.ndarray = np.array ( [beta for _ in range ( 0, epoch, step )] )

	figure, axis = plt.subplots ()
	axis.plot ( epochs, trainingLoss, label = "Training Loss", color = "blue" )
	plotTrendLine ( axis, epochs, trainingLoss, color = 'blue' )
	axis.plot ( epochs, validationLoss, label = "Validation Loss", color = 'yellow' )
	plotTrendLine ( axis, epochs, validationLoss, color = 'yellow' )

	subtitle: str = "Beta: {:03} v{} Model #{:02} Epochs {}".format ( beta, Config.version,
	                                                                  Config.getArgs ().model_number,
	                                                                  epoch )
	# Don't include the beta if our losses are very small.
	if beta // 4 <= trainingLoss[0]:
		axis.plot ( epochs, betas, 'r--,', label = "Beta", )
		subtitle = subtitle[10:]

	axis.set ( xlabel = 'Epochs', ylabel = 'Loss' )
	plt.title ( subtitle, fontsize = 10 )
	plt.suptitle ( "Training and Validation Loss" )
	axis.legend ()
	axis.grid ()

	figure.savefig (
		"plots/loss-v{}-E{:04}-N{:02}.png".format ( Config.version, epoch, Config.getArgs ().model_number ) )
	plt.show ()


def plotDifferences ():
	bigDictionary: typing.Dict = train.loadModel ()

	epoch: int = bigDictionary["startingEpoch"]
	step: int = min ( epoch // 25 + 1, 10 )

	trainingDiff: typing.List[typing.Tuple[float, float]] = bigDictionary["trainingDifference"][::step]
	validationDiff: typing.List[typing.Tuple[float, float]] = bigDictionary["validationDifference"][::step]

	trainingDiffPos: typing.List[float] = [diff[0] for diff in trainingDiff]
	validationDiffPos: typing.List[float] = [diff[0] for diff in validationDiff]

	trainingDiffOri: typing.List[float] = [diff[1] for diff in trainingDiff]
	validationDiffOri: typing.List[float] = [diff[1] for diff in validationDiff]

	epochs: np.ndarray = np.arange ( 0, epoch, step )

	figure, (axisPos, axisOri) = plt.subplots ( 2, 1, sharex = True )

	axisPos.plot ( epochs, trainingDiffPos, label = "Training", color = 'blue' )
	plotTrendLine ( axisPos, epochs, trainingDiffPos, color = 'blue' )
	axisPos.plot ( epochs, validationDiffPos, label = "Validation", color = 'yellow' )
	plotTrendLine ( axisPos, epochs, validationDiffPos, color = 'yellow' )
	axisPos.set ( ylabel = "Position Difference (m)" )
	plt.suptitle ( "Training and Validation Difference" )

	fontdict: typing.Dict = { 'fontsize':            10,
	                          'fontweight':          matplotlib.rcParams['axes.titleweight'],
	                          'verticalalignment':   'baseline',
	                          'horizontalalignment': "center"
	                          }

	axisPos.set_title ( "v{} Model #{:02} Epoch {}".format ( Config.version, Config.getArgs ().model_number, epoch ),
	                    fontdict = fontdict )
	axisPos.legend ()

	axisOri.plot ( epochs, trainingDiffOri, label = "Training", color = 'blue' )
	plotTrendLine ( axisOri, epochs, trainingDiffOri, color = 'blue' )
	axisOri.plot ( epochs, validationDiffOri, label = "Validation", color = 'yellow' )
	plotTrendLine ( axisOri, epochs, validationDiffOri, color = 'yellow' )
	axisOri.set ( xlabel = "Epochs", ylabel = "Orientation Difference" )
	axisOri.legend ()

	figure.savefig (
		"plots/diff-v{}-E{:04}-N{:02}.png".format ( Config.version, epoch, Config.getArgs ().model_number ),
		bbox_inches = "tight",
		pad_inches = .2 )
	plt.show ()


def plotTestData ():
	bigDictionary: typing.Dict = train.loadModel ()

	epoch: int = bigDictionary["startingEpoch"]

	step: int = min ( 3, epoch // 25 + 1 )

	testingDiff: typing.List[typing.Tuple[float, float]] = bigDictionary["testingDifference"][::step]
	uncertainty: typing.List[typing.Tuple[float, float]] = bigDictionary["uncertainty"][::step]
	anees: typing.List[typing.Tuple[float, float]] = bigDictionary["anees"][::step]

	diffPos: typing.List[float] = [diff[0] for diff in testingDiff]
	certPos: typing.List[float] = [cert[0] for cert in uncertainty]

	diffOri: typing.List[float] = [diff[1] for diff in testingDiff]
	certOri: typing.List[float] = [cert[1] for cert in uncertainty]

	aneesPos: typing.List[float] = [err[0] for err in anees]
	aneesOri: typing.List[float] = [err[1] for err in anees]

	epochs: np.ndarray = np.arange ( 0, epoch, epoch / len ( testingDiff ) )

	figure, axis = plt.subplots ( 2, 3, sharex = True )

	axisPos: np.ndarray = axis[0]
	axisOri: np.ndarray = axis[1]

	axisDiffPos: Axes = axisPos[0]
	axisDiffOri: Axes = axisOri[0]
	axisCertPos: Axes = axisPos[1]
	axisCertOri: Axes = axisOri[1]
	axisANEESPos: Axes = axisPos[2]
	axisANEESOri: Axes = axisOri[2]

	axisDiffPos.plot ( epochs, diffPos, color = 'blue' )
	plotTrendLine ( axisDiffPos, epochs, diffPos, color = 'blue' )
	axisDiffPos.set ( ylabel = "Position Difference (m)" )
	plt.suptitle ( "v{} Model #{:02} Epoch {}".format ( Config.version, Config.getArgs ().model_number, epoch ) )

	axisDiffPos.set_title ( "Testing Difference" )

	axisDiffOri.plot ( epochs, diffOri, color = 'green' )
	plotTrendLine ( axisDiffOri, epochs, diffOri, color = 'green' )
	axisDiffOri.set ( xlabel = "Epochs", ylabel = "Orientation Difference" )

	axisCertPos.plot ( epochs, certPos, color = 'blue' )
	plotTrendLine ( axisCertPos, epochs, certPos, color = 'blue' )
	axisCertPos.set ( ylabel = "Position Uncertainty" )
	axisCertPos.set_title ( "Testing Uncertainty" )

	axisCertOri.plot ( epochs, certOri, color = 'green' )
	plotTrendLine ( axisCertOri, epochs, certOri, color = 'green' )
	axisCertOri.set ( xlabel = "Epochs", ylabel = "Orientation Uncertainty" )

	axisANEESPos.plot ( epochs, aneesPos, color = 'blue' )
	plotTrendLine ( axisANEESPos, epochs, aneesPos, color = 'blue' )
	axisANEESPos.set ( ylabel = "ANEES (Position)" )
	axisANEESPos.set_title ( "ANEES" )

	axisANEESOri.plot ( epochs, aneesOri, color = 'green' )
	plotTrendLine ( axisANEESOri, epochs, aneesOri, color = 'green' )
	axisANEESOri.set ( ylabel = "ANEES (Orientation)" )

	# wspace is space between each subplot in the width
	# while hspace is space between each subplot in the height.
	figure.subplots_adjust ( wspace = 0.8, hspace = 0.3 )

	figure.savefig (
		'plots/test-v{}-E{:04}-N{:02}.png'.format ( Config.version, epoch, Config.getArgs ().model_number ),
		bbox_inches = "tight",
		pad_inches = .4 )
	plt.show ()


def plotOutput ():
	bigDictionary = train.loadModel ()
	network: torch.nn.Module = bigDictionary["network"]
	epoch: int = bigDictionary["startingEpoch"]

	# Load data
	image: Image = Image.open ( Config.getArgs ().image )
	transforms = tv.transforms.Compose ( [
		# Resize the image to 256x256
		tv.transforms.Resize ( (256, 256) ),
		# Crop the image to 224 x 224
		tv.transforms.RandomCrop ( (224, 224) ),
		tv.transforms.ToTensor ()
	] )

	output, outputs, uncertPos, uncertOri = train.testSingleImage ( network = network, image = image,
	                                                                transforms = transforms )

	def getPos ( theta ):
		return np.array ( [np.cos ( theta ), np.sin ( theta )] )

	# The real position of the network.
	realPos = output[:2]
	# The different positions from dropout.
	positions = outputs[:, :2]
	# The uncertainty matrix for positions.
	certPos = uncertPos[:2, :2]
	# The value of uncertainty for position
	certPosValue = certPos.diagonal ().sum ()

	U, S, _ = np.linalg.svd ( certPos )

	fig = plt.figure ( 0 )
	axis = fig.add_subplot ( 111, aspect = 'equal' )

	for pos in positions:
		axis.scatter ( pos[0], pos[1] )

	precision = 500
	theta = np.linspace ( 0.0, 2.0 * np.pi, precision )

	positionForEllipse = U * np.sqrt ( S ) @ getPos ( theta )

	axis.scatter ( realPos[0], realPos[1], c = '000' )

	for pos in positionForEllipse.T:
		axis.scatter ( pos[0] + realPos[0], pos[1] + realPos[1], c = '000' )

	phases = ["train", "validate", "test"]
	for phase in phases:
		data = dataset.PoseNetDataSet ( transform = transforms, phase = phase,
		                                dataroot = Config.getArgs ().database_root )
		label: np.ndarray = data[Config.getArgs ().image]
		if label is not None:
			break
	if label is None:
		Logger.warn ( "Could not find label for given image. Plotting without it." )
	else:
		actualPos = label[:2]
		axis.scatter ( actualPos[0], actualPos[1], marker = '*' )
		errorVector = np.subtract( realPos, actualPos )
		anees = train.ANEES ( errorVector.reshape(1, 2), certPos.reshape ( 1, 2, 2 ), 1 )
		bottom, top, left, right, height, width = getBounds ( axis = axis )
		center = ( ( left + right ) / 2.0, ( bottom + top ) / 2.0 )
		# Make numpy print 2 decimal digits.
		np.set_printoptions(formatter={'float_kind': lambda x: "{:.2}".format(x)}, suppress = True)
		data_text = "Uncert: {:.2} Error: {} ANEES: {:.2}".format ( certPosValue, errorVector, anees )
		# Reset print options to their default value.
		np.set_printoptions()
		font_size = 10
		axis.text ( center[0], bottom - height / 2, data_text, fontsize = font_size, horizontalalignment = 'center', verticalalignment = 'bottom' )

	plt.suptitle ( "Uncertainty In Detail" )
	plt.title ( "v{} E{:04} #{:02}".format ( Config.version, epoch, Config.getArgs ().model_number ) )

	axis.update_datalim ( positionForEllipse.T )
	axis.grid ( True )

	plt.show ()

	fig.savefig (
		"plots/uncert-v{}-E{:04}-N{:02}.png".format ( Config.version, epoch, Config.getArgs ().model_number ) )


def main ():
	plotLosses ()
	plotDifferences ()
	plotTestData ()
