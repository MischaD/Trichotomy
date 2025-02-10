import argparse
import json
import os
from agent_siamese_network import AgentSiameseNetwork


# define an argument parser
parser = argparse.ArgumentParser('Patient Verification')
parser.add_argument('--config_path', default='./', help='the path where the config files are stored')
parser.add_argument('--config', default='config.json', help='the hyper-parameter configuration and experiment settings')
args = parser.parse_args()
print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)

# read config
with open(args.config_path + args.config, 'r') as config:
    config = config.read()

# parse config
config = json.loads(config)

# create folder to save experiment-related files
os.makedirs(os.path.join('./archive/' , config['experiment_description']), exist_ok=True)
SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'

# call agent and run experiment
experiment = AgentSiameseNetwork(config)
experiment.run()