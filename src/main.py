from utils import tab_printer
from parser import parameters 
from parser import parameter_parsers #if env command line use it
from capsgnn import CapsGNNTrainer

#Create json data from root directory
#saveAsJSON(Randomsamples = 1, realsamples = 1)

#gets the parameters
args = parameters()
#args = parameters()
#prints the current paramanter config
tab_printer(args)
#creates the model base on parameters
model = CapsGNNTrainer(args)
#fits on training data
model.fit()
#validates on test data
model.score()
#saves validations
model.save_predictions()
print('')
#predicts on given graph path
print(model.predict('/home/eguray/Desktop/CapsGNN-torch/input/test/69.json'))
print('')
#saves current model
model.save_model()
print('')
#loads the given model
model.load_model('20190708-130347.pth')
print('')
#predicts on given graph path
print(model.predict('/home/eguray/Desktop/CapsGNN-torch/input/test/69.json'))
print('')
model.load_model('test.pth')
print('')
#predicts on given graph path
print(model.predict('/home/eguray/Desktop/CapsGNN-torch/input/test/69.json'))
