import plot
import train
import util
from config import Config
from logger import Logger
# main function

if __name__ == '__main__':
	args = Config.getArgs ()
	util.printHeader ()
	if args.plot:
		plot.main ()
	elif args.image is None:
		Logger.log ( "Going to train network!" )
		train.main ()
	else:
		plot.plotOutput ()
