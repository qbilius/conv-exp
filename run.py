from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse, sys, os
import numpy as np

import matplotlib
matplotlib.use('Agg')  # making sure plotting works on clusters
import seaborn as sns

import base

np.random.seed(0)

global_tasks = ['report', 'download_datasets', 'download_models', 'compute_features']
datasets = set([d for d in os.listdir('.') if os.path.isdir(d) and d != '.git'] + global_tasks)

parser = argparse.ArgumentParser()
parser.add_argument('exp', choices=datasets, help='Choose which dataset to use')
parser.add_argument('task', nargs='?', choices=['run', 'compare', 'report'], default='run')
parser.add_argument('func', nargs='?', default='run')
parser.add_argument('--subset')
parser.add_argument('-m', '--model', default='caffenet')
                    #choices=models.KNOWN_MODELS)
parser.add_argument('-p', '--model_path', default=None)
parser.add_argument('--layers', default='all')
parser.add_argument('-d', '--dry', action='store_true')
parser.add_argument('--savefig', default='', choices=['svg','png'])
# parser.add_argument('--saveresps', action='store_true')
parser.add_argument('-f', '--force', action='store_true')
parser.add_argument('--forcemodels', action='store_true')
parser.add_argument('--filter', action='store_true')
parser.add_argument('--mode', default='gpu', help='Caffe mode: cpu or gpu (default)')
parser.add_argument('-c', '--context', default='paper',
                    choices=['paper', 'notebook', 'talk', 'poster'])
parser.add_argument('-b', '--bootstrap', action='store_true', help='Whether you want to bootstrap (no flag means false).')
parser.add_argument('--dissim', default='correlation', help='Dissimilarity metric (default: correlation)')
args = parser.parse_args()

sns.set_context(args.context)
sns.set_style('white')
sns.set_palette('Set2')

if args.layers == 'None':
    layers = None
else:
    try:
        layers = eval(args.layers)
    except:
        layers = args.layers
    else:
        if not isinstance(layers, (tuple, list)):
            layers = args.layers

# if args.exp in ['geons'] and args.subset is None and args.task != 'report':
#     raise Exception('For geons dataset you must choose a subset '
#                     'using the --subset flag.')

if args.forcemodels: args.force = True

kwargs = {'exp': args.exp,
          'task': args.task,
          'func': args.func,
          'subset': args.subset,
          'model_name': args.model,
          'model_path': args.model_path,
          'layers': layers,
          'mode': args.mode,
          'savedata': not args.dry,
          'force': args.force,
          'forcemodels': args.forcemodels,
          'filter': args.filter,
          'savefig': args.savefig,
          'report': False,
          'bootstrap': args.bootstrap,
          'html': None,
          'dissim': args.dissim}

if args.exp in global_tasks:
    kwargs['task'] = args.exp
    base.run(**kwargs)
else:
    base.run(**kwargs)
