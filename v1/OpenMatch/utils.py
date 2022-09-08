import os
import json
from argparse import Action

class DictOrStr(Action):
    def __call__(self, parser, namespace, values, option_string=None):
         if '=' in values:
             my_dict = {}
             for kv in values.split(","):
                 k,v = kv.split("=")
                 my_dict[k] = v
             setattr(namespace, self.dest, my_dict)
         else:
             setattr(namespace, self.dest, values)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id+' Q0 '+str(value[0])+' '+str(rank+1)+' '+str(value[1][0])+' openmatch\n')
    return

def save_features(rst_file, features):
    with open(rst_file, 'w') as writer:
        for feature in features:
            writer.write(feature+'\n')
    return
