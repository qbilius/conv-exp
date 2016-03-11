from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, glob
import shutil, io, urllib, zipfile  # for downloading datasets
import pickle
from collections import OrderedDict

import numpy as np
import scipy.stats
import pandas

import seaborn as sns

try:
    from nltk.corpus import wordnet as wn
except:
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet as wn

from psychopy_ext import models, stats, plot, report, utils


SHALLOW = ['px', 'gaborjet', 'hog', 'phog', 'phow']
HMAX = ['hmax99', 'hmax_hmin', 'hmax_pnas']#, 'randfilt']
DEEP = ['caffenet', 'vgg-19', 'googlenet']
ALL_EXPS = ['snodgrass', 'hop2008', 'fonts', 'geons', 'stefania']

COLORS = sns.color_palette('Set2', 8)
ftemplate = '{exp}/{kind}_{exp}_{name}.pkl'

pref2value = {'corr': 'correlation',
              'pred_corr': 'confidence0',
              'acc': 'accuracy',
              'dis': 'dissimilarity',
              'clust': 'dissimilarity',
              'dis_group': 'similarity'
              }
pref2func = {'corr': 'corr',
             'pred_corr': 'pred_corr',
             'acc': 'accuracy',
             'dis': 'dissimilarity',
             'clust': 'cluster',
             'dis_group': 'dis_group',
             'dis_group_diff': 'dis_group_diff'}

def load_image(im, resize=None, *args, **kwargs):
    return utils.load_image(im, resize=(256,256), *args, **kwargs)

# models.Model.load_image = load_image
models.NICE_NAMES['hmo'] = 'HMO'


def get_data(pref):
    def decorator(func):
        def func_wrapper(self):
            if self.forcemodels:
                return func(self)
            elif self.force:
                if pref not in ['resps', 'dis', 'preds'] and self.model_name not in self.dims:
                    return func(self)
                else:
                    data = self.load(pref)
                    if data is None:
                        return func(self)
                        # raise Exception('model {} not recognized'.format(self.model_name))
            else:
                data = self.load(pref)
                if data is None:
                    return func(self)
            if self.task == 'run' and self.func is not None and pref not in ['resps','dis']:
                if self.func.startswith(pref):
                    self.plot_single(data, pref)
            return data
        return func_wrapper
    return decorator

def style_plot(plot_func):
    def func_wrapper(self, df, values, ylim=[0,1]):
        g = plot_func(self, df, values, ylim=ylim)

        if 'ci_low' in df.columns:
            hue = 'kind' if 'kind' in df else None
            plot.plot_ci(df, hue=hue)

        if 'accuracy' in df.columns:
            for value in self.dims.values():
                plot_chance(value)

        if self.model_name in ['googlenet', 'vgg-19']:
            labels = g.axes.flat[0].get_xticklabels()
            for label in labels:
                if len(label.get_text()) > 3:
                    label.set_ha('right')
                    label.set_rotation(30)
            # if self.model_name == 'googlenet':
            #     sns.plt.subplots_adjust(bottom=.25)
        self.show(values)
    return func_wrapper

def plot_chance(value):
    chance = 1. / len(np.unique(value))
    sns.plt.axhline(chance, ls='--', c='.15')

def msg(*args):
    if len(args) == 1:
        print('{}'.format(*args))
    elif len(args) == 2:
        print('{}: {}'.format(*args))
    else:
        print(args)

def load(pref='', exp='', subset=None, suffix='', layers=None,
         filt_layers=True):
    if suffix in ['shape','category','human']:
        path = 'data'
    elif suffix == 'hmo' and pref == 'resps':
        path = 'data'
    else:
        path = 'computed'

    path = os.path.join(exp, path)
    name = '_'.join(filter(None, [pref, exp, subset, suffix]))
    name = os.path.join(path, name)
    try:
        data = pickle.load(open(name+'.pkl', 'rb'))
    except:
        msg('could not load from', name+'.pkl')
        return None
        # try:
        #     data = scipy.io.loadmat(open(name+'.mat', 'rb'))
        # except:
        # else:
        #     msg('loaded from', name+'.mat')
        #     data = OrderedDict([(model_name, data[model_name])])
    else:
        msg('loaded from', name + '.pkl')

    if filt_layers and pref != 'preds':
        data = filter_layers(data, layers)
    return data

def save(data, pref='', exp='', subset=None, suffix='', savedata=True,
         ext='pkl'):
    path = os.path.join(exp, 'computed')
    name =  '_'.join(filter(None, [pref, exp, subset, suffix])) + '.' + ext
    name = os.path.join(path, name)
    if savedata:
        if not os.path.isdir(path): os.makedirs(path)
        pickle.dump(data, open(name, 'wb'))
        msg('saved to', name)

def show(pref='', exp='', subset=None, suffix='', savefig='', html=None,
         caption=None):
    name =  '_'.join(filter(None, ['plot', pref, exp, subset, suffix]))

    if html is not None:
        html.writeimg(name, caption=caption)
        # path = os.path.join(html.path, html.imgdir)
    else:
        name += '.' + savefig
        path = os.path.join(exp, 'computed')
        if not os.path.isdir(path): os.makedirs(path)
        name = os.path.join(path, name)
        if savefig != '':
            sns.plt.savefig(name, dpi=300)
            msg('saved to', name)
        else:
            sns.plt.show()

def filter_layers(data, layers):

    if isinstance(data, dict):
        avail_layers = data.keys()
    elif isinstance(data, pandas.DataFrame):
        avail_layers = data.layer.unique()
    else:
        raise ValueError('Data type not recognized', type(data))
    msg('available layers', ', '.join(avail_layers))

    if layers in [None, 'top', 'output']:
        layers = [avail_layers[-1]]
    elif layers == 'all':
        layers = avail_layers
        msg('WARNING: you requested all layers; make sure these are all')
    elif isinstance(layers, str):
        try:
            data = OrderedDict([(layers, data[layers])])
        except:
            msg('layers not found, reclassifying', layers)
            return None
    elif isinstance(layers, int):
        layers = [avail_layers[layers]]
    else:
        if not all([l in avail_layers for l in layers]):
            msg('not all requested layers found, reclassifying', layers)
            return None

    if isinstance(data, dict):
        data = OrderedDict([(layer, data[layer]) for layer in layers])
    elif isinstance(data, pandas.DataFrame):
        data = data[data.layer.isin(layers)]
    msg('using layers', ', '.join(layers))

    return data

def row2dis(row):
    n = int((1 + np.sqrt(1 + 8*row.shape[1])) / 2)
    dis = np.zeros((row.shape[0], n, n))
    inds = np.triu_indices(n, k=1)
    dis[:,inds[0],inds[1]] = row
    dis = np.rollaxis(dis, 1, 3)
    dis[:,inds[0],inds[1]] = row
    for d in dis: np.fill_diagonal(d, np.nan)
    return dis

def dis2row(dis):
    inds = np.triu_indices(dis.shape[1], k=1)
    if dis.ndim == 3:
        row = [d[inds] for d in dis]
    else:
        row = dis[inds]
    return row


class Base(object):

    def __init__(self, model_path=None, layers='all',
                 exp='', subset=None, mode='gpu',
                 savedata=True, savefig='', saveresps=False,
                 force=False, forcemodels=False, filter=False, task=None,
                 func=None, report=None, bootstrap=False, html=None,
                 skip_hmo=True, dissim='correlation'):

        self.exp = exp
        self.savedata = savedata
        self.savefig = savefig
        self.saveresps = saveresps
        self.force = force
        self.forcemodels = forcemodels
        self.filter = filter
        self.task = task
        self.func = func
        self.report = report
        self.html = html
        self.bootstrap = bootstrap
        self.mode = mode
        self.layers = layers
        self.skip_hmo = skip_hmo
        self.dissim = dissim

        self.set_models()
        self.set_subset(subset)

    def download_dataset(self, url=None, path='', ext=None):
        """Downloads and extract datasets
        """
        print('Downloading and extracting data...')
        r = urllib.urlopen(url)
        namelist = []
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            if ext is not None:
                fnames = z.namelist()
                for fname in fnames:
                    if os.path.splitext(fname)[1] == ext:
                        source = z.open(fname)
                        new_path = os.path.join(path, os.path.basename(fname))
                        target = file(new_path, 'wb')
                        with source, target:
                            shutil.copyfileobj(source, target)
                        namelist.append(new_path)
            else:
                z.extractall(path)
                namelist = z.namelist()
        return namelist

    def set_model(self, model_name):
        if model_name in models.ALIASES:
            self.model_name = models.ALIASES[model_name]
        else:
            self.model_name = model_name

    def __getattr__(self, name):
        if name == 'ims':
            if 'ims' not in self.__dict__:
                return self._get_ims()
            else:
                if len(self.__dict__['ims']) == 0:
                    return self._get_ims()
                else:
                    return self.__dict__[name]
        else:
            return self.__dict__[name]

    def _get_ims(self):
        if 'impath' not in self.__dict__:
            self.set_subset()
        ims = sorted(glob.glob(self.impath))
        if len(ims) == 0:
            self.get_images()
            ims = sorted(glob.glob(self.impath))
        return ims

    def set_subset(self, subset=None):
        if subset is not None:
            self.impath = os.path.join(self.exp, 'img', subset, '*.*')
        else:
            self.impath = os.path.join(self.exp, 'img', '*.*')
        self.subset = subset

    def get_images(self):
        raise NotImplemented

    def set_models(self):
        if self.skip_hmo:
            deep = [d for d in DEEP if d!='hmo']
        else:
            deep = DEEP

        models = [('shallow',m) for m in SHALLOW]
        models += [('hmax',m) for m in HMAX]
        models += [('deep',m) for m in deep]
        self.models = models

    def load(self, pref):
        print()
        print('{:=^50}'.format(' ' + self.model_name + ' '))

        if self.filter and pref != 'dis':
            subset = 'filt' if self.subset is None else self.subset + '_filt'
        else:
            subset = self.subset

        return load(pref=pref, exp=self.exp, subset=subset,
                    suffix=self.model_name, layers=self.layers)

    def save(self, data, pref):
        if self.filter:
            subset = 'filt' if self.subset is None else self.subset + '_filt'
        else:
            subset = self.subset
        save(data, pref=pref, exp=self.exp, subset=subset,
             suffix=self.model_name, savedata=self.savedata)

    def show(self, pref, suffix=None, caption=None):
        if suffix is None: suffix = self.model_name
        if self.filter:
            subset = 'filt' if self.subset is None else self.subset + '_filt'
        else:
            subset = self.subset
        return show(pref=pref, exp=self.exp, subset=subset,
                    suffix=suffix, savefig=self.savefig,
                    html=self.html, caption=caption)

    def synsets_from_csv(self, fname, sep=','):
        with open(fname, 'rb') as f:
            lines = f.readlines()
        df = []
        for line in lines:
            spl = line.strip('\n').split(sep)
            try:
                synset = wn._synset_from_pos_and_offset(spl[0][0], int(spl[0][1:]))
            except:
                import pdb; pdb.set_trace()

            df.append({'id':spl[0], 'names':spl[1], 'synset':synset})
        # df = pandas.DataFrame(df, columns=['id', 'names', 'synset'])
        return df

    def synsets_from_txt(self, fname):
        with open(fname, 'rb') as f:
            lines = f.readlines()
        df = []
        for line in lines:
            w = line.split()[0]
            descr = line.strip('\r\n').replace(w+' ', '')
            synset = wn._synset_from_pos_and_offset(w[0], int(w[1:]))
            df.append({'id':w, 'names':descr, 'synset':synset})
        # df = pandas.DataFrame(df, columns=['id', 'names', 'synset'])
        return df

    @get_data('resps')
    def classify(self):
        try:
            m = models.get_model(self.model_name)
        except:
            msg('%s is not available for generating responses' %self.model_name)
            resps = self.load('resps')
            if resps is None:
                raise ValueError('no response file found for %s' %
                                 self.model_name)
        else:
            m.load_image = load_image
            output = m.run(self.ims, layers=self.layers, return_dict=True)
            resps = OrderedDict()
            for layer, out in output.items():
                resps[layer] = out.reshape((out.shape[0], -1))

            if self.model_name in ['hmax_hmin', 'hmax_pnas']:
                self.save(resps, 'resps')
        return resps

    @get_data('preds')
    def predict(self):
        try:
            m = models.get_model(self.model_name)
        except:
            msg('%s is not available for generating responses' %self.model_name)
            raise Exception

            # resps = self.load('resps')
            # if resps is None:
            #     raise ValueError('no response file found for %s' %
            #                     self.model_name)
        else:
            m.load_image = load_image
            preds = m.predict(self.ims, topn=5)
            self.save(preds, 'preds')
        return preds

    def pred_acc(self, compute_acc=True):
        if compute_acc:
            preds = self.predict()
        imagenet_labels = self.synsets_from_txt('synset_words.txt')
        dataset_labels = self.synsets_from_csv(os.path.join(self.exp, 'data', self.exp + '.csv'))
        all_hyps = lambda s:s.hyponyms()

        df = pandas.DataFrame.from_dict(dataset_labels)
        df['imgid'] = ''
        df['imdnames'] = ''
        df['kind'] = 'unknown'
        df['accuracy'] = np.nan
        df['accuracy0'] = np.nan
        df['confidence0'] = np.nan
        for no, dtlab in enumerate(dataset_labels):
            hypos = set([i for i in dtlab['synset'].closure(all_hyps)])
            hypos = hypos.union([dtlab['synset']])
            for imglab in imagenet_labels:
                if imglab['synset'] in hypos:
                    df.loc[no, 'imgid'] = imglab['id']
                    df.loc[no, 'imgnames'] = imglab['names']
                    if imglab['id'] == df.loc[no, 'id']:
                        df.loc[no, 'kind'] = 'exact'
                    else:
                        df.loc[no, 'kind'] = 'superordinate'
                    break
            if compute_acc:
                acc = False
                acc0 = False
                for i,p in enumerate(preds[no]):
                    psyn = wn._synset_from_pos_and_offset(p['synset'][0],
                                                          int(p['synset'][1:]))
                    df.loc[no, 'pred%d'%i] = ', '.join(psyn.lemma_names())
                    # check if the prediction is exact
                    # or at least more specific than the correct resp
                    if psyn in hypos:
                        acc = True
                    if i==0:
                        if psyn in hypos:
                            acc0 = True
                if acc == False:
                    if df.loc[no, 'kind'] != 'unknown':
                        df.loc[no, 'accuracy'] = False
                else:
                    df.loc[no, 'accuracy'] = True
                if acc0 == False:
                    if df.loc[no, 'kind'] != 'unknown':
                        df.loc[no, 'accuracy0'] = False
                else:
                    df.loc[no, 'accuracy0'] = True
                df.loc[no, 'confidence0'] = preds[no][0]['confidence']
        return df

    @get_data('dis')
    def dissimilarity(self):
        resps = self.classify()
        dis = models.dissimilarity(resps, kind=self.dissim)
        self.save(dis, 'dis')
        return dis

    @get_data('clust')
    def cluster(self):
        resps = self.classify()
        clust = []
        dims = []
        for dim, labels in self.dims.items():
            out = models.cluster(resps, labels=labels, bootstrap=self.bootstrap, niter=1000)
            clust.append(out)
            dims.extend([dim]*len(out))

        clust = pandas.concat(clust, axis=0)
        clust.insert(0, 'kind', dims)

        self.save(clust, 'clust')
        if self.task == 'run':
            self.plot_single(clust, 'clust')
        return clust

    def cluster_behav(self):
        return None

    def mds(self, icons=None, seed=None, **kwargs):
        if self.model_name in self.dims:
            dim = self.model_name
            dis = self.dissimilarity()
            # dis = load(pref='dis', exp=self.exp, suffix=dim)
            if dis[dim].ndim == 3:
                dis[dim] = np.mean(dis[dim], axis=0)
        else:
            dis = self.dissimilarity()

        if icons is None: icons = self.ims
        names = [os.path.splitext(os.path.basename(im))[0] for im in self.ims]
        mds_res = models.mds(dis)
        models.plot_data(mds_res, kind='mds', icons=icons, **kwargs)
        self.show('mds')

    @get_data('corr')
    def corr(self):
        dis = self.dissimilarity()
        df = []
        nname = models.NICE_NAMES[self.model_name].lower()
        for dim in self.dims:
            dim_data = load(pref='dis', exp=self.exp, suffix=dim)
            if dim_data is None:
                name = self.model_name
                self.set_model(dim)
                dim_data = self.dissimilarity()
                self.set_model(name)
                if dim_data is None:
                    raise Exception('dimension data %s cannot be obtained' % dim)

            dim_data = dim_data[dim]
            if dim_data.ndim == 3:
                dim_data = np.mean(dim_data, axis=0)
            struct = self.dims[dim] if self.exp in ['fonts', 'stefania'] else None
            if self.filter:
                dim_data = dim_data[self.sel][:,self.sel]
                struct = None
            for layer, data in dis.items():
                d = data[self.sel][:,self.sel] if self.filter else data
                corr = stats.corr(d, dim_data, sel='upper')
                if self.bootstrap:
                    print('bootstrapping stats...')
                    bf = stats.bootstrap_resample(d, dim_data,
                            func=stats.corr, ci=None, seed=0, sel='upper',
                            struct=struct)
                    for i, b in enumerate(bf):
                        df.append([dim, nname, layer, corr, i, b])
                else:
                    df.append([dim, nname, layer, corr, 0, np.nan])
        df = pandas.DataFrame(df, columns=['kind', 'models', 'layer',
                                            'correlation', 'iter', 'bootstrap'])
        self.save(df, pref='corr')
        if self.task == 'run':
            self.plot_single(df, 'corr')
        return df

    # def plot_corr(self, subplots=False, **kwargs):
    #     self.corr = self.corr.rename(columns={'model1': 'kind',
    #                          'model2': '%s layer' %self.model_name})
    #     self.plot_single_model(self.corr, subplots=subplots, **kwargs)
    #     plot_ci(self.corr)

    def reliability(self):
        rels = OrderedDict()
        for dim in self.dims:
            self.set_model(dim)
            data = load(pref='dis', exp=self.exp, suffix=dim)[dim]
            if data.ndim == 3:
                if self.filter:
                    inds = np.triu_indices(data[0][self.sel][:,self.sel].shape[1], k=1)
                    df = np.array([d[self.sel][:,self.sel][inds] for d in data])
                else:
                    inds = np.triu_indices(data.shape[1], k=1)
                    df = np.array([d[inds] for d in data])
                rels[dim] =  stats.reliability(df)
            else:
                rels[dim] = [np.nan, np.nan]
        return rels

    @style_plot
    def plot_single(self, df, pref, ylim=[0,1]):
        if len(self.dims) == 1:
            hue = None
            color = self.colors.values()[0]
            palette = None
        else:
            hue = 'kind'
            color = None
            palette = [self.colors[dim] for dim in self.dims]

        g = sns.factorplot('layer', pref2value[pref], data=df, hue=hue, ci=0, kind='point', color=color, palette=palette, aspect=2)

        dff = _set_ci(df, groupby=['kind', 'layer'])

        palette = [self.colors[dim] for dim in self.dims]
        for kind, col in zip(dff.kind.unique(), palette):
            sel = dff.kind == kind
            sns.plt.fill_between(range(len(dff[sel].layer.unique())),
                dff[sel].ci_low, dff[sel].ci_high, zorder=0,
                color=col, alpha=.3)
        # sns.plt.ylim([-.1, 1.1])
        sns.plt.ylim(ylim)
        sns.plt.title(models.NICE_NAMES[self.model_name])

        return g

class Compare(object):

    def __init__(self, myexp):
        self.myexp = myexp

    def classify(self):
        for depth, model_name in self.myexp.models:
            self.myexp.set_model(model_name)
            self.myexp.classify()

    def dissimilarity(self):
        for depth, model_name in self.myexp.models:
            self.myexp.set_model(model_name)
            self.myexp.dissimilarity()

    def predict(self, clear_memory=False):
        for depth, model_name in self.myexp.models:
            if depth == 'deep':
                self.myexp.set_model(model_name)
                self.myexp.predict()

    def load(self, pref):
        print()
        print('{:=^50}'.format(' ' + pref2func[pref] + ' '))
        if self.myexp.filter and pref != 'dis':
            subset = 'filt' if self.myexp.subset is None else self.myexp.subset + '_filt'
        else:
            subset = self.myexp.subset
        return load(pref=pref, exp=self.myexp.exp, subset=subset,
                    suffix='all', filt_layers=False)

    def save(self, data, pref):
        if self.myexp.filter:
            subset = 'filt' if self.myexp.subset is None else self.myexp.subset + '_filt'
        else:
            subset = self.myexp.subset

        save(data, pref=pref, exp=self.myexp.exp, subset=subset,
             suffix='all', savedata=self.myexp.savedata)

    def show(self, pref, suffix='all'):
        if self.myexp.filter:
            subset = 'filt' if self.myexp.subset is None else self.myexp.subset + '_filt'
        else:
            subset = self.myexp.subset
        # import pdb; pdb.set_trace()
        return show(pref=pref, exp=self.myexp.exp, suffix=suffix,
                    subset=subset,
                    savefig=self.myexp.savefig, html=self.myexp.html)

    def get_data_all(self, pref, kind, **kwargs):
        # if force:
        #     import pdb; pdb.set_trace()
        #
        #     df = getattr(self, '_' + pref + '_all')()
        # else:
        # if not self.myexp.force:
        #     df = self.load(pref)
        #     if df is None: df = getattr(self, '_' + kind + '_all')(pref)
        # else:
        df = getattr(self, '_' + kind + '_all')(pref, **kwargs)

        if pref not in ['preds', 'pred_corr']:
            dfs =[filter_layers(df[df.models==m], self.myexp.layers) for m in df.models.unique()]
            df = pandas.concat(dfs, ignore_index=True)
        return df

    def compare(self, pref, ylim=[-.1,1]):
        print()
        print('{:=^50}'.format(' ' + pref + ' '))
        df = self.get_data_all(pref, kind='compare')
        if hasattr(self.myexp, 'behav'):
            behav = self.myexp.behav()
        else:
            behav = None

        if behav is not None:
            rels = {'shape':stats.bootstrap_resample(behav.dissimilarity, func=np.mean)}
        else:
            rels = None

        if pref == 'dis_group_diff':
            values = 'preference for perceived shape'
            df = df.rename(columns={'preference': values})
            self.plot_all(df, values, 'diff', pref=pref, ceiling=None, color=self.myexp.colors['shape'], ylim=ylim)
        elif pref == 'pred_corr':
            values = 'correlation'
            df['kind'] = 'shape'
            # df = df.rename(columns={'preference': values})
            behav = self.myexp.behav()
            behav = behav.pivot_table(index=['kind', 'subjid'],
                                      columns='no', values='acc')
            # for subset in df.dataset.unique():
            #     self.myexp.set_subset(subset)
            #     rel = stats.reliability(behav.loc[subset])
            #     rel = ((1+rel[0])/2., (1+rel[1])/2.)
            self.plot_all(df, values, 'consistency', col='dataset', pref=pref, ceiling=None, color=self.myexp.colors['shape'], ylim=ylim)
        else:
            if self.myexp.exp == 'fonts':
                values = 'clustering accuracy'
                df = df.rename(columns={'dissimilarity': values})
            else:
                values = 'accuracy'
            for dim in self.myexp.dims:
                ceiling = None if rels is None else rels[dim]
                self.plot_all(df[df.kind==dim], values, dim, pref=pref, ceiling=ceiling, color=self.myexp.colors[dim], ylim=ylim)

        if self.myexp.bootstrap:
            bf = self.bootstrap_ttest_grouped(df)
            if self.myexp.bootstrap:
                if self.myexp.html is not None:
                    self.myexp.html.writetable(bf,
                        caption='bootstrapped t-test (one-tailed, rel. samples)')

    def _compare_all(self, pref, **kwargs):
        df = []
        props = []
        for depth, model_name in self.myexp.models:
            self.myexp.set_model(model_name)
            out = getattr(self.myexp, pref2func[pref])(**kwargs)
            df.append(out)
            name = models.NICE_NAMES[model_name].lower()
            props.extend([[depth,name]] * len(out))
        df = pandas.concat(df, axis=0, ignore_index=True)

        props = np.array(props).T
        df.insert(0, 'depth', props[0])
        df.insert(1, 'models', props[1])

        # self.save(df, pref=pref)
        return df

    def corr(self):
        print()
        print('{:=^50}'.format(' corr '))
        self.layers = 'output'
        msg('WARNING', 'using only the output layer')
        df = self.get_data_all('corr', kind='corr')
        rels = self.myexp.reliability()
        for dim in self.myexp.dims:
            self.plot_all(df[df.kind==dim], 'correlation', dim, pref='corr', ceiling=rels[dim], color=self.myexp.colors[dim])
            if self.myexp.bootstrap:
                bf = self.bootstrap_ttest_grouped(df[df.kind==dim])
                if self.myexp.html is not None:
                    self.myexp.html.writetable(bf,
                        caption='bootstrapped t-test (one-tailed, rel. samples)')
        # df = pandas.concat(dfs, axis=0)
        # import pdb; pdb.set_trace()

    def _corr_all(self, pref):
        df = []
        props = []
        for depth, model_name in self.myexp.models:
            self.myexp.set_model(model_name)
            out = self.myexp.corr()
            df.append(out)
            props.extend([depth] * len(out))

        df = pandas.concat(df, axis=0, ignore_index=True)
        props = np.array(props).T
        df.insert(0, 'depth', props)
        # df.insert(1, 'models', props[1])
        # self.save(df, pref=pref)
        return df

    def _corr_all_orig(self, pref):
        df = []
        for dim in self.myexp.dims:
            dim_data = load(pref='dis', exp=self.myexp.exp, suffix=dim)[dim]
            if dim_data.ndim == 3:
                dim_data = np.mean(dim_data, axis=0)
            for depth, model_name in self.myexp.models:
                self.myexp.set_model(model_name)
                dis = self.myexp.dissimilarity()
                layer = dis.keys()[-1]
                dis = dis[layer]
                corr = stats.corr(dis, dim_data, sel='upper')
                if self.myexp.bootstrap:
                    print('bootstrapping stats...')
                    bf = stats.bootstrap_resample(dis, dim_data, func=stats.corr, ci=None, seed=0, sel='upper',
                        struct=self.dims[dim].ravel())
                    for i, b in enumerate(bf):
                        df.append([dim, depth, model_name, layer, corr, i, b])
                else:
                    df.append([dim, depth, model_name, layer, corr, 0, np.nan])
        df = pandas.DataFrame(df, columns=['kind', 'depth', 'models', 'layer',
                                           'correlation', 'iter', 'bootstrap'])
        self.save(df, pref=pref)
        return df

    def plot_all(self, df, values, dim, pref='', ceiling=None, color=None, ylim=[-.1, 1.1]):

        df = _set_ci(df)
        print(df)

        gray = sns.color_palette('Set2', 8)[-1]
        light = (.3,.3,.3)#colors[-2]
        # if color is None:
        #     color = self.colors[0]
        # light = sns.light_palette(color, n_colors=3)[1]
        palette = []
        for model in df.models.unique():
            depth = df[df.models==model].depth.iloc[0]
            if depth == 'shallow':
                palette.append(gray)
            elif depth == 'hmax':
                palette.append(light)
            elif depth == 'deep':
                palette.append(color)
        # dims = df.kind.unique()
        # col = None if len(dims) == 1 else 'kind'
        g = sns.factorplot(x='models', y=values,
                            data=df, kind='bar', palette=palette)

        sns.plt.ylim(ylim)
        if 'ci_low' in df.columns:
            hue = 'kind' if 'kind' in df else None
            plot.plot_ci(df, hue=hue)
        if 'accuracy' in df.columns and self.myexp.dims[dim] is not None:
            plot_chance(self.myexp.dims[dim])
        # for ax, dim in zip(g.axes.flat, dims):
        sns.plt.title(dim)
        if ceiling is not None:
            sns.plt.axhspan(ceiling[0], ceiling[1], facecolor='0.9', edgecolor='0.9', zorder=0)

        self.set_vertical_labels(g)
        #pref = kind if pref == '' else pref + '_' + kind
        self.show(pref=pref, suffix='all_' + dim)
        # if self.html is not None:
        #     self.html.writetable(df)
    def set_vertical_labels(self, g):
        for ax in g.axes.flat:
            for n, model in enumerate(ax.get_xticklabels()):
                ax.text(n, .03, model.get_text(), rotation='vertical',
                        ha='center', va='bottom', backgroundcolor=(1,1,1,.5))
            ax.set_xticklabels([])

    def bootstrap_ttest_grouped(self, bf, tails='one'):
        bfg = bf.groupby(['depth', 'iter']).mean()
        bfg = bfg.unstack(level='depth').bootstrap
        st = []
        for nd1,d1 in enumerate(bfg):
            for d2 in bfg.iloc[:,nd1+1:]:
                diff = np.squeeze(bfg[d1].values - bfg[d2].values)
                pct = scipy.stats.percentileofscore(diff, 0, kind='mean') / 100.
                p = min(pct, 1-pct)
                if tails == 'two': p *= 2
                star = stats.get_star(p)
                st.append([d1, d2, np.mean(diff), p, star])
        st = pandas.DataFrame(st, columns=['depth1', 'depth2', 'mean', 'p', 'sig'])
        print(st)
        return st

def _set_ci(df, groupby=['kind', 'models'], ci=95):
    f_low = lambda x: np.percentile(x, 50-ci/2.)
    f_high = lambda x: np.percentile(x, 50+ci/2.)

    df = stats.factorize(df)
    pct = df.groupby(groupby).bootstrap.agg({'ci_low': f_low, 'ci_high':f_high}).reset_index()
    df = df.groupby(groupby).agg(lambda x: x.iloc[0]).reset_index()
    df['ci_low'] = pct.ci_low
    df['ci_high'] = pct.ci_high
    return df

def gen_report(model_name=None, **kwargs):
    kwargs['func'] = None
    kwargs['report'] = True
    kwargs['savefig'] = 'svg'
    # kwargs['bootstrap'] = True
    mod = __import__(kwargs['exp']+'.run', fromlist=[kwargs['exp'].title()])
    getattr(mod, 'report')(**kwargs)

def get_exp(model_name=None, **kwargs):
    mod = __import__(kwargs['exp']+'.run', fromlist=[kwargs['exp'].title()])
    if kwargs['exp'] == 'hop2008':
        c = 'HOP2008'
    elif kwargs['exp'] == '2ndorder':
        c = 'SecondOrder'
    else :
        c = kwargs['exp'].title()
    Exp = getattr(mod, c)
    return mod, Exp(**kwargs)

def run(model_name, **kwargs):
    reppath='report/'
    if kwargs['exp'] in ['download_datasets', 'compute_features']:
        for exp in ALL_EXPS:
            print()
            print('#' * 80)
            print('{:^80s}'.format(' ' + exp + ' '))
            print('#' * 80)
            print()
            kwargs['exp'] = exp
            mod, myexp = get_exp(**kwargs)

            if kwargs['task'] == 'download_datasets':
                myexp.set_model(model_name)
                getattr(myexp, 'get_images')()
            elif kwargs['task'] == 'compute_features':
                c = getattr(mod, 'Compare')
                if exp == 'snodgrass':
                    getattr(c(myexp), 'predict')()
                else:
                    getattr(c(myexp), 'dissimilarity')()
    elif kwargs['exp'] == 'download_models':
        for model in SHALLOW + HMAX:
            models.Model(model).download_model()
    elif kwargs['exp'] == 'report':
        html = report.Report(path=reppath, imgext='svg')
        html.open()
        for exp in ALL_EXPS:
            print()
            print('#' * 80)
            print('{:^80s}'.format(' ' + exp + ' '))
            print('#' * 80)
            print()
            kwargs['exp'] = exp
            kwargs['html'] = html
            html.imgdir = exp
            gen_report(**kwargs)
        html.close()
    elif kwargs['task'] == 'report':
        html = report.Report(path=reppath, imgdir=kwargs['exp'])
        kwargs['html'] = html
        html.open()
        gen_report(**kwargs)
        html.close()
    elif kwargs['task'] == 'compare':
        mod, myexp = get_exp(**kwargs)
        c = getattr(mod, 'Compare')
        getattr(c(myexp), kwargs['func'])()
    else:
        mod, myexp = get_exp(**kwargs)
        myexp.set_model(model_name)
        getattr(myexp, kwargs['func'])()
