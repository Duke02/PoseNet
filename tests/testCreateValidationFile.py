import util

util.createValidationDataFile ( filenameTrain = "KingsCollege/dataset_train.txt",
                                filenameValidate = "KingsCollege/dataset_validate.txt" )

validationFile = open ( "KingsCollege/dataset_validate.txt", "r" )
lines = validationFile.readlines ()
print ( lines[0] )
print ( lines[-1] )
validationFile.close ()
