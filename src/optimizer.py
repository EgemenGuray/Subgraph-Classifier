import logging
logging.basicConfig(level=logging.WARNING)

import argparse

from utils import CapsGNNWorker

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

        
worker = CapsGNNWorker(run_id='0')
cs = worker.get_configspace()

config = cs.sample_configuration().get_dictionary()
res = worker.compute(config=config, budget=20, working_directory='.')    


