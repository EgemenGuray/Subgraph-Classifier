import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

import logging
logging.basicConfig(level=logging.DEBUG)
import argparse


from utils import tab_printer
from parser import parameters 
from parser import parameter_parsers #if env command line use it
from capsgnn import CapsGNNTrainer


class CapsGNNWorker(Worker):
    
    def __init__(self, N_train =8192, N_valid = 1024, **kwargs):
        
        super().__init__(**kwargs)
    
    """
    It builds the configuration space with the needed hyperparameters.
    It is easily possible to implement different types of hyperparameters.
    Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
    returns ConfigurationsSpace-Object
    """
    
    @staticmethod
    def get_configspace():
            
            
            import os
            cwd = os.getcwd()[:-4]

            cs = CS.ConfigurationSpace()
            epochs = CSH.CategoricalHyperparameter('epochs', [100])
            test_graph_folder = CSH.CategoricalHyperparameter('test_graph_folder', [cwd + '/input/test/'])
            inner_attention_dimension = CSH.UniformIntegerHyperparameter('inner_attention_dimension', lower=8, upper=40, default_value=20, log=True)
            lambd = CSH.UniformFloatHyperparameter('lambd', lower=0.4, upper=0.6, default_value=0.5, log=True)
            theta = CSH.UniformFloatHyperparameter('theta', lower=0.001, upper=0.002, default_value=0.0016, log=True)
            prediction_path = CSH.CategoricalHyperparameter('prediction_path', [cwd + '/output/predictions.csv'])
            learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.0001, upper=0.1, default_value=0.001, log=True)
            gcn_layers = CSH.UniformIntegerHyperparameter('gcn_layers', lower=2, upper=32, default_value=2, log=True)
            batch_size = CSH.CategoricalHyperparameter('batch_size', [9,15])
            train_graph_folder = CSH.CategoricalHyperparameter('train_graph_folder', [cwd +  '/input/train/'])
            weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=1e-1, default_value=1e-2, log=True)
            number_of_capsules = CSH.UniformIntegerHyperparameter('number_of_capsules', lower=2, upper=32, default_value=8, log=True)
            gcn_filters = CSH.UniformIntegerHyperparameter('gcn_filters', lower=8, upper=48, default_value=20, log=True)
            capsule_dimensions = CSH.UniformIntegerHyperparameter('capsule_dimensions', lower=2, upper=16, default_value=8, log=True)

            cs.add_hyperparameters([epochs,
                                    test_graph_folder, 
                                    inner_attention_dimension, 
                                    lambd, 
                                    theta,	
                                    prediction_path, 
                                    learning_rate, 
                                    gcn_layers,	
                                    batch_size,	
                                    train_graph_folder,	
                                    weight_decay, 
                                    number_of_capsules, 
                                    gcn_filters, 
                                    capsule_dimensions
                                    ])

            return cs
            
       
    def compute(self, config, budget, working_directory, *args, **kwargs):
        
        
        par = argparse.Namespace(**config)
        #prints the current paramanter config
        tab_printer(par)
        #creates the model base on parameters
        model = CapsGNNTrainer(par)
        #fits on training data
        model.fit()
        #validates on test data
        validation_accuracy = model.score()
        #saves validations
        return ({
                    'loss': 1-validation_accuracy, # remember: HpBandSter always minimizes!
                    'info': config

            })

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=9)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=243)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=4)
args=parser.parse_args()        

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()


# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = CapsGNNWorker(nameserver='127.0.0.1',run_id='example1')
w.run(background=True)


# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.eguray
# The run method will return the `Result` that contains all runs performed.
bohb = BOHB(  configspace = w.get_configspace(),
              run_id = 'example1', nameserver='127.0.0.1',
              min_budget=args.min_budget, max_budget=args.max_budget
           )
res = bohb.run(n_iterations=args.n_iterations)


# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()


# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))
