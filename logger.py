import typing
from datetime import datetime

from config import Config


class Logger:
	# The levels of configuration for logging.
	logConfig: typing.Dict[str, int] = { "all":  1,
	                                     "main": 0,
	                                     "min":  -1,
	                                     "warn": -2,
	                                     "err":  -3,
	                                     "none": -4 }

	@staticmethod
	def log ( message: str, logger: str = "all" ) -> None:
		"""
		Prints log statements with a time stamp. Only prints if the logger is less than the log
		configuration provided by the user's arguments.

		:param message: The message to be printed
		:param logger: The verbosity of the log statement.
		"""
		# If we can log based on the configuration
		if Logger.shouldLog ( logger = logger ):
			# Then log.
			print ( str ( datetime.now () ) + ": " + str ( message ) )

	@staticmethod
	def shouldLog ( logger: str = "all" ) -> bool:
		"""

		:param logger: The verbosity of the log.
		:return: Whether the log should be printed.
		"""
		return Logger.logConfig[logger] <= Logger.logConfig[Config.getArgs ().log_config]

	@staticmethod
	def error ( message: str ) -> None:
		"""
		Uses the log statement to print an Error message (a log with ERROR: prefixed to the log message.).
		Is always printed unless the user sets --log-config to be none.

		:param message: The error message to be printed.
		"""
		Logger.log ( "ERROR: " + str ( message ), logger = "err" )

	@staticmethod
	def warn ( message: str ) -> None:
		"""
		Uses the log statement to print a warning message (a log with WARNING: prefixed to the log message).
		Is always printed unless the user sets --log-config to none or err

		:param message: The warning to be printed.
		"""
		Logger.log ( "WARNING: " + str ( message ), logger = "warn" )
