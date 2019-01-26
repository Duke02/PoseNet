import posenet
import util

model = util.getPretrainedModel ( path = "./pretrained-models/places-googlenet.pickle" )
print ( model.keys () )
network = posenet.PoseNet ( input_nc = 3, weights = model )
