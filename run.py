import argparse, sys, os
import numpy as np
import seaborn as sns

sys.path.insert(0, '../psychopy_ext')
from psychopy_ext import models

np.random.seed(0)

# detect all datasets
# datasets = {}
# for root, folders, files in os.walk('.'):
#     if root[-3:] == 'img':
#         droot = root.replace('\\', '/').split('/')[1]
#         if len(folders) == 0:
#             datasets[droot] = root
#         else:
#             for folder in folders:
#                 dname = '_'.join([droot, folder])
#                 datasets[dname] = os.path.join(root, folder)



datasets = [d for d in os.listdir('.') if os.path.isdir(d)]

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=datasets)
parser.add_argument('task', choices=['run', 'compare'])
parser.add_argument('func', default='run')
parser.add_argument('--subset')
parser.add_argument('-m', '--model', default='CaffeNet')
                    #choices=models.KNOWN_MODELS)
parser.add_argument('--layers', default='all')
parser.add_argument('-d', '--dry', action='store_true')
parser.add_argument('--savefig', default='', choices=['svg','png'])
# parser.add_argument('--saveresps', action='store_true')
parser.add_argument('-f', '--force', action='store_true')
parser.add_argument('--filter', action='store_true')
parser.add_argument('--mode', default='gpu')
parser.add_argument('-c', '--context', default='paper',
                    choices=['paper', 'notebook', 'talk', 'poster'])
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

if args.dataset in ['geons'] and args.subset is None:
    raise Exception('For geons dataset you must choose a subset '
                    'using the --subset flag.')

kwargs = {'task': args.task,
          'func': args.func,
          'dataset': args.dataset, 'subset': args.subset,
          'model_name': args.model,
          'layers': layers,
          'mode': args.mode,
          'savedata': not args.dry,
          'force': args.force,
          'filter': args.filter,
          'savefig': args.savefig}

os.chdir(kwargs['dataset'])
mod = __import__(kwargs['dataset'] + '.run')
if args.task == 'run':
    getattr(mod.run, 'run')(**kwargs)
else:
    getattr(mod.run, args.func)(**kwargs)
