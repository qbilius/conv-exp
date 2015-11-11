import argparse, sys, os
import numpy as np
import seaborn as sns

sys.path.insert(0, '../psychopy_ext')
from psychopy_ext import models, stats, plot, report
import base2

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


datasets = [d for d in os.listdir('.') if os.path.isdir(d)] + ['report']

parser = argparse.ArgumentParser()
parser.add_argument('exp', choices=datasets)
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
parser.add_argument('--forceresps', action='store_true')
parser.add_argument('--filter', action='store_true')
parser.add_argument('--mode', default='gpu')
parser.add_argument('-c', '--context', default='paper',
                    choices=['paper', 'notebook', 'talk', 'poster'])
parser.add_argument('-b', '--bootstrap', action='store_true')
parser.add_argument('--dissim', default='correlation')
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

if args.exp in ['geons'] and args.subset is None:
    raise Exception('For geons dataset you must choose a subset '
                    'using the --subset flag.')

if args.forceresps: args.force = True

kwargs = {'task': args.task,
          'func': args.func,
          'exp': args.exp,
          'subset': args.subset,
          'model_name': args.model,
          'model_path': args.model_path,
          'layers': layers,
          'mode': args.mode,
          'savedata': not args.dry,
          'force': args.force,
          'forceresps': args.forceresps,
          'filter': args.filter,
          'savefig': args.savefig,
          'report': False,
          'bootstrap': args.bootstrap,
          'html': None,
          'dissim': args.dissim}

if args.exp == 'report':
    kwargs['task'] = 'report'
    base2.run(**kwargs)
else:
    base2.run(**kwargs)
#elif args.task == 'run':
#    m = base2.Base(**kwargs)
#    getattr(m, args.func)()

# def run(args, **kwargs):
#     path = os.path.dirname(os.path.abspath(__file__))
#     print path
#     os.chdir(os.path.join(path, kwargs['dataset']))
#     mod = __import__(kwargs['dataset'] + '.run')
#
#     if args.task in ['run', 'compare']:
#         getattr(mod.run, 'run')(**kwargs)
#     else:
#         getattr(mod.run, args.func)(**kwargs)
#
#
# if kwargs['dataset'] == 'report':
#     kwargs['task'] = 'compare'
#     kwargs['func'] = 'report'
#     html, kwargs = base.html(kwargs, path='report/')
#     html.open()
#     for exp in ['fonts']:#['snodgrass', 'hop2008', 'fonts', 'geons', 'stefania']:
#         html.writeh(exp, h='h1')
#         os.chdir(exp)
#         kwargs['dataset'] = exp
#         mod = __import__(exp + '.run')
#         getattr(mod.run, 'report')(**kwargs)
#     html.close()
#
# else:
#     run(args, **kwargs)
