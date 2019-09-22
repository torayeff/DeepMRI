import sys
sys.path.append('/home/agajan/DeepMRI')
from deepmri import utils  # noqa: E402
from DiffusionMRI.bkpmodels.bkp3.ConvModel5 import Encoder, Decoder  # noqa: E402  # noqa: E402

encoder = Encoder(input_size=(145, 145))
decoder = Decoder()

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))