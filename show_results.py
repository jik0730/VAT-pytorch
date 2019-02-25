import os
import json
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument(
    '--parent_dir',
    default='experiments/embedding_omni',
    help='Directory containing params.json')
parser.add_argument('--tag', default='val', help='val or best')


def iterate_parent_dir(parent_dir, tag):
    results = OrderedDict()
    for dirname in os.listdir(parent_dir):
        child_dir = os.path.join(parent_dir, dirname)
        if os.path.isdir(child_dir):
            for filename in os.listdir(child_dir):
                if filename == 'results.json':
                    jsonname = os.path.join(child_dir, filename)
                    results[dirname] = read_acc_from_json(jsonname)
    for key in sorted(results.keys()):
        print(key, results[key])


def read_acc_from_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return float(data['Best ' + args.tag + ' score'])


if __name__ == '__main__':
    args = parser.parse_args()
    iterate_parent_dir(args.parent_dir, args.tag)
