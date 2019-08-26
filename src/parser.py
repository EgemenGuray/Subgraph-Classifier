import argparse
import os
cwd = os.getcwd()[:-4]

def parameters(epochs : int = 100, 
					test_graph_folder = cwd + '/input/test/', 
					inner_attention_dimension : int = 20,
					lambd :  float = 0.5, 
					theta :  float = 0.1, 
					prediction_path = cwd + '/output/oss.csv', 
					learning_rate : float = 0.01, 
					gcn_layers : int = 2,
					batch_size : int = 32,
					train_graph_folder = cwd +  '/input/train/',
					weight_decay : float= 0.16,
					number_of_capsules : int = 8,
					gcn_filters : int = 20,
					capsule_dimensions : int = 8
					):
    
    ret_dict = {
    	'epochs': epochs, 
    	'test_graph_folder': test_graph_folder, 
    	'inner_attention_dimension': inner_attention_dimension, 
    	'lambd': lambd, 
    	'theta': theta, 
    	'prediction_path': prediction_path, 
    	'learning_rate': learning_rate, 
    	'gcn_layers': gcn_layers,
    	'batch_size': batch_size,
    	'train_graph_folder': train_graph_folder,
    	'weight_decay': weight_decay,
    	'number_of_capsules': number_of_capsules,
    	'gcn_filters': gcn_filters,
    	'capsule_dimensions': capsule_dimensions
        }
	
    ret_val = argparse.Namespace(**ret_dict)
    
    return ret_val


"""
Parses command line parameters. By default it learns on the watts-strogatz dataset
The default hyperparameters give good results without cross-validation 
"""
def parameter_parsers():
    
    parser = argparse.ArgumentParser(description = "Run CapsGNN.")
	
    parser.add_argument("--train-graph-folder",
                        nargs = "?",
                        default = cwd +  '/input/train/',
	                help = "Training graphs folder.")

    parser.add_argument("--test-graph-folder",
                        nargs = "?",
                        default = cwd + '/input/test/',
	                help = "Testing graphs folder.")

    parser.add_argument("--prediction-path",
                        nargs = "?",
                        default = cwd + '/output/predictions.csv',
	                help = "Path to store the predicted graph labels.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 100,
	                help = "Number of training epochs. More epochs better for complex dataset. Default is 100.")

    parser.add_argument("--batch-size",
                        type = int,
                        default = 32,
	                help = "Number of graphs processed per batch. Smaller batches better for learning but increases the time to learn. Default is 32.")

    parser.add_argument("--gcn-filters",
                        type = int,
                        default = 20,
	                help = "Number of Graph Convolutional filters. Default is 20.")

    parser.add_argument("--gcn-layers",
                        type = int,
                        default = 2,
	                help = "Number of Graph Convolutional Layers. More layers helps to represent more features in forms of node embeddings, but makes model to harder to learn. Default is 2.")

    parser.add_argument("--inner-attention-dimension",
                        type = int,
                        default = 20,
	                help = "Number of Attention Neurons. Default is 20.")

    parser.add_argument("--capsule-dimensions",
                        type = int,
                        default = 8,
	                help = "Capsule dimensions. Default is 8.")

    parser.add_argument("--number-of-capsules",
                        type = int,
                        default = 8,
	                help = "Number of capsules per layer. Default is 8.")

    parser.add_argument("--weight-decay",
                        type = float,
                        default = 10**-6,
	                help = "Weight decay. Default is 10^-6.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Sets the amount of descent on loss vs value of weight function during gradient descent. Higher values overshoots minimum, smaller values increases the number of steps needed to descent. Default is 0.01.")

    parser.add_argument("--lambd",
                        type = float,
                        default = 0.5,
	                help = "Loss combination weight. Default is 0.5.")

    parser.add_argument("--theta",
                        type = float,
                        default = 0.1,
	                help = "Reconstruction loss weight. Weight for penalizg the GCN layer for creating capsule forms different from the input. Helps model to capture important parts, omit irrelevant parts. Default is 0.1.")
    
    return parser.parse_args()
