from config import Config
from posenet import PoseNet
from torch import save
from logger import Logger

fileName = Config.getArgs ().model_file.format ( "test", 0 )
network = PoseNet (input_nc = 3)
try:
	save ( obj = { "model": network.state_dict () }, f = fileName )
except FileNotFoundError:
	Logger.log ( "ERROR: Can not find file.", logger = "min" )
	Logger.log ( "\tCould it be that your working directory doesn't have the directory you specified?",
	      logger = "min" )
Logger.log ( "Saving was successful.", logger = "min" )
