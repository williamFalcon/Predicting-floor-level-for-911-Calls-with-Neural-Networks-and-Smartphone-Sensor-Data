import pandas as pd
from test_tube import HyperOptArgumentParser
import os

"""
Finds the max metric requested from an experiment (across versions)

Example:
python find_max.py --exp_path /Users/waf/Dropbox/crackerjack/floor/test_tube/test_tube_data/final_1 --metric val_acc

"""
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--exp_path', type=str)
parser.add_argument('--metric', type=str)
hyperparams = parser.parse_args()

max_score = 0
results = {}
for folder in os.listdir(hyperparams.exp_path):
    if '.DS' in folder:
        continue

    file_path = hyperparams.exp_path + '/' + folder + '/metrics.csv'
    df = pd.read_csv(file_path)
    max_val = df[hyperparams.metric].max()
    if max_val > max_score:
        max_score = max_val
        results = {hyperparams.metric : max_val, 'folder': folder}

print(results)