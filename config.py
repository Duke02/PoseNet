import argparse
import typing

import torch


class Config:
	# The arguments from the user
	args: typing.Union[None, argparse.Namespace] = None
	version: str = '1.6'

	@staticmethod
	def generateArgs () -> argparse.Namespace:
		"""
			This function parses and returns the arguments provided by the user.

			Use python main.py --help to get a full list of arguments.

			:return: Returns the parsed arguments given by the user.
			"""
		# Argument parser. Most defaults are what the original paper outlined.
		arg_parser = argparse.ArgumentParser ()

		# Groups for the arguments
		systemArgs = arg_parser.add_argument_group ( 'system',
		                                             "Args that affect the system to be used during the running of the program." )
		trainingArgs = arg_parser.add_argument_group ( 'training', "Args that affect training." )
		loggingArgs = arg_parser.add_argument_group ( 'logging', "Args that affect logging." )
		fileArgs = arg_parser.add_argument_group ( 'files', "Args that deal with files (saving and loading)." )
		testingArgs = arg_parser.add_argument_group ( 'testing', "Args that deal with testing." )
		actionArgs = arg_parser.add_argument_group ( 'actions', "Args that deal with what the program does." )

		# Add flag argument to run with GPU or not.
		systemArgs.add_argument ( '-g', '--gpu',
		                          action = 'store_true',
		                          required = False,
		                          help = "Use GPU",
		                          default = False )
		systemArgs.add_argument ( '--version',
		                          action = 'version',
		                          dest = 'version',
		                          version = Config.version )
		systemArgs.add_argument ( '-t', '--threads',
		                          help = "The number of threads that you want to use.",
		                          type = int,
		                          required = False,
		                          default = 1 )
		systemArgs.add_argument ( '-T', '--max-threads',
		                          help = "Use the most number of threads possible.",
		                          action = "store_true",
		                          required = False,
		                          default = False )
		systemArgs.add_argument ( '--force-threads',
		                          help = "Force the program to not limit the number of threads from 1-7",
		                          action = "store_true",
		                          required = False,
		                          default = False )
		systemArgs.add_argument ( '--model-number',
		                          help = "The number of the model. To be used when saving.",
		                          type = int,
		                          required = False,
		                          default = 0 )

		# Reduce learning rate by this rate
		# the gamma in the LR scheduler
		trainingArgs.add_argument ( '-F', '--factor',
		                            help = "Reduce learning rate by factor",
		                            type = float,
		                            required = False,
		                            default = .1 )
		# The base learning rate to start out with.
		trainingArgs.add_argument ( '-l', '--learning-rate',
		                            help = "Standard learning rate",
		                            type = float,
		                            required = False,
		                            default = 1e-05 )
		# The momentum for the network.
		trainingArgs.add_argument ( '-m', '--momentum',
		                            help = "Momentum rate",
		                            type = float,
		                            required = False,
		                            default = .9 )
		# Batch size for the network.
		trainingArgs.add_argument ( '-b', '--batch-size',
		                            help = "Batch size",
		                            type = int,
		                            required = False,
		                            default = 75 )
		# Beta for the loss function.
		trainingArgs.add_argument ( '-B', '--beta',
		                            help = "Beta for loss function",
		                            type = int,
		                            required = False,
		                            default = 500 )
		trainingArgs.add_argument ( '-e', '--epochs',
		                            help = "Total number of epochs for this model",
		                            type = int,
		                            required = False,
		                            default = 10 )
		trainingArgs.add_argument ( '-d', '--database-root',
		                            type = str,
		                            help = "The root folder of the database to be used.",
		                            required = False,
		                            default = "KingsCollege/" )
		trainingArgs.add_argument ( '--threshold-factor',
		                            help = "When loss is less than the beta times this number, halve the threshold. Should be (0,1]",
		                            required = False,
		                            type = float,
		                            default = 2.0 / 3.0 )

		loggingArgs.add_argument ( '-v', '--verbose',
		                           help = "Print everything the neural network is doing.",
		                           action = 'store_true',
		                           required = False,
		                           default = False )
		# Print progress every nth batch.
		loggingArgs.add_argument ( '-p', '--print-every',
		                           help = "Print progress every nth batch",
		                           type = int,
		                           required = False,
		                           default = 4 )
		loggingArgs.add_argument ( '-L', '--log-config',
		                           help = "How much the program should log.",
		                           type = str,
		                           choices = ["all", "main", "min", "warn", "err", "none"],
		                           required = False,
		                           default = "main" )

		fileArgs.add_argument ( '-f', "--model-file",
		                        help = "Save model to this file",
		                        type = str,
		                        required = False,
		                        default = "models/posenet-model-v{}-E{:04d}-N{:02d}.model" )
		# Use the provided pretrained model.
		fileArgs.add_argument ( '-M', '--pretrained-model',
		                        help = "Resume using given pretrained model",
		                        type = str,
		                        required = False,
		                        default = None )
		fileArgs.add_argument ( '-s', '--dont-save',
		                        help = "Don't save models after each epoch. Default action is to save models after each epoch.",
		                        required = False,
		                        action = 'store_true',
		                        default = False )
		fileArgs.add_argument ( '-r', '--resume',
		                        help = "Resume from latest model",
		                        action = 'store_true',
		                        required = False,
		                        default = False )

		testingArgs.add_argument ( '--num-of-tests',
		                           help = "Number of times to test the network to get the uncertainty.",
		                           required = False,
		                           default = 64,
		                           type = int )
		testingArgs.add_argument ( '--test-every',
		                           help = "Test every given epochs.",
		                           required = False,
		                           default = 2,
		                           type = int )
		testingArgs.add_argument ( '--test-print-lots',
		                           help = "Print testing results with the frequency based on the batch-size (default) or not. "
		                                  + "\nWith the batch size, testing results will be printed less.",
		                           required = False,
		                           action = "store_true",
		                           default = False )

		actionArgs.add_argument ( '-i', '--image',
		                          type = str,
		                          help = "A single image to test the network on.",
		                          required = False,
		                          default = None )
		actionArgs.add_argument ( '--skip-training',
		                          help = "Skips training and validation and goes straight to testing. Good to use if you keep getting memory errors.",
		                          required = False,
		                          action = "store_true",
		                          default = False )
		actionArgs.add_argument ( '--plot',
		                          help = "Plot losses of specified model.",
		                          required = False,
		                          action = "store_true",
		                          default = False )
		actionArgs.add_argument ( '--skip-testing',
		                          help = "Skip testing (useful if you get memory errors only while testing.)",
		                          required = False,
		                          action = "store_true",
		                          default = False )

		out: argparse.Namespace = arg_parser.parse_args ()
		if out.verbose:
			out.log_config = "all"
		if out.resume and out.pretrained_model is None:
			out.pretrained_model = "models/posenet-latest-v{version}-N{num:02d}.model".format (
					version = Config.version, num = out.model_number )

		if out.max_threads or out.threads > torch.get_num_threads ():
			out.threads = torch.get_num_threads ()
		if out.threads >= 8 and not out.force_threads:
			out.threads = 7
		if out.threads is 0:
			out.threads = 1

		return out

	@staticmethod
	def getArgs () -> argparse.Namespace:
		"""
		Returns args. If args is None, then it generates args.

		:return: args based on user input.
		"""
		# Generate arguments if we need to
		if Config.args is None:
			Config.args = Config.generateArgs ()
		return Config.args

	# Function that sees if we have a GPU and if we want to use the GPU.
	# If we do, use them.
	@staticmethod
	def useCuda () -> bool:
		"""
		A helper function to see if the program should use CUDA.

		:return: True if the computer has CUDNN and the user passed the GPU flag.
		"""
		return torch.cuda.is_available () and Config.getArgs ().gpu

	@staticmethod
	def useParallelData () -> bool:
		"""
		A helper function to see if the program should use data parallelism on its GPUs.

		:return: true if it should.
		"""

		return Config.useCuda () and Config.getArgs ().threads > 1

	@staticmethod
	def getDevice () -> str:
		if Config.useCuda ():
			return 'cuda:{}'.format ( torch.cuda.current_device () )
		return 'cpu'
