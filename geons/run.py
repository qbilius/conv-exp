import os, sys
from collections import OrderedDict

import numpy as np
import scipy.stats
import pandas
import seaborn as sns

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import models, stats

import base


class Geons(base.Base):

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.dims = OrderedDict([('nap', [0,1])])
        self.colors = OrderedDict([('nap', base.COLORS[1])])
        self.skip_hmo = True
        super(Geons, self).__init__(*args, **kwargs)

    @base.get_data('nap')
    def dissimilarity(self):
        resps = self.classify()
        df = []
        dims = get_dims()
        for layer, resps in resps.items():
            for g,i in enumerate(range(0, len(resps), 3)):
                dis = models.dissimilarity(resps[i:i+3],
                                                      kind='euclidean')
                n = os.path.basename(self.ims[i])[:2]
                df.append([layer, g, int(n), dims[n], 'non-accidental', dis[1,0]])
                df.append([layer, g, int(n), dims[n], 'metric', dis[1,2]])

        nap = pandas.DataFrame(df,
                            columns=['layer', 'geon', 'fno', 'dimension', 'kind', 'dist'])
        self.save(nap, 'nap')
        return nap

    def accuracy(self, plot=True):
        nap = self.dissimilarity()

        acc = nap.pivot_table(values='dist', index=['layer', 'geon', 'fno', 'dimension'], columns='kind').reset_index()
        acc['accuracy'] = acc['non-accidental'] > acc['metric']

        if self.bootstrap:
            dfs = []
            for layer in acc.layer.unique():
                sel = acc[acc.layer==layer]['accuracy']
                pct = stats.bootstrap_resample(sel, ci=None, func=np.mean)
                d = OrderedDict([('kind', ['nap'] * len(pct)),
                                 ('layer', [layer]*len(pct)),
                                 ('accuracy', sel.mean()),
                                 ('iter', range(len(pct))),
                                 ('bootstrap', pct)])
                dfs.append(pandas.DataFrame.from_dict(d))
            df = pandas.concat(dfs)

        else:
            df = acc.groupby('layer').mean().reset_index()
            df['kind'] = 'nap'
            df['iter'] = 0
            df['bootstrap'] = np.nan

        if self.task == 'run' and plot:
            self.plot_single(df, 'acc')
        return df


    def plot_acc(self):
        xlabel = '%s layer' % self.model_name
        self.acc = self.acc.rename(columns={'layer': xlabel})
        # self.acc['kind'] = 'nap'
        self.dims = OrderedDict([('nap', [0,1])])
        orange = sns.color_palette('Set2')[1]
        self.plot_single_model(self.acc, subplots=False,
                               colors=orange)
        self.show(pref='acc')

    def make_html(ims, resps, pref=''):

        h = markup.page( )
        h.init(title='{} results'.format(self.dataset, self.model_name),
               css='style.css')
        h.table()
        h.tr()
        h.td('NA')
        h.td('base')
        h.td('metric')
        h.tr.close()

        for rno, row in enumerate(resps):
            h.tr()
            ind = np.argmax(row)
            if ind == 0:
                h.td(class_='gray')
            else:
                h.td()
            h.img(src=ims[rno*3], width='100px')
            h.p('%.2f' % row[0])
            h.p.close()
            h.td.close()

            h.td()
            h.img(src=ims[rno*3+1], width='100px')
            h.p('--')
            h.p.close()
            h.td.close()

            if ind == 1:
                h.td(class_='gray')
            else:
                h.td()
            h.img(src=ims[rno*3+2], width='100px')
            h.p('%.2f' % row[1])
            h.p.close()
            h.td.close()

            h.tr.close()

        h.table.close()
        mn = np.argmax(resps, axis=1)
        h.h1('Total performance: %d%%' % (np.sum(mn==0)*100./len(mn)))
        h.h1.close()
        with open('results_%s.html' % pref,'wb') as f:
            f.write('\n'.join(h.content))

def get_dims():
    with open('geons/data/dimensions.csv', 'rb') as f:
        lines = f.readlines()
    dims = dict([l.strip('\n').split(',') for l in lines])
    return dims

def remake_amir2012(**kwargs):
    data = scipy.io.loadmat('amir2012.mat')
    meta = data['Results'][0,0]
    acc = meta['CRmat']
    ims = data['ImageNames']
    df = []
    variants = {0: 'metric', 1: 'non-accidental'}
    vobs = {0: 'variant', 1: 'base'}
    versions = {0: '3d', 1: '2d'}
    dims = get_dims()

    df = []
    for (imno, varno, run, subjid, vob, verno), value in np.ndenumerate(acc):
        impath = ims[imno,0,0]

        if len(impath) != 0:
            cond = int(impath[0].split('/')[-1].split('os')[0])
            dim = dims['%02d' % cond]
            df.append([meta['Subjects'][0,subjid][0],
                       run,
                       variants[varno],
                       versions[verno],
                       vobs[vob],
                       cond,
                       dim,
                       meta['RTmat'][imno, varno, run, subjid, vob, verno],
                       value
                        ])
    df = pandas.DataFrame(df, columns=['subjid', 'run', 'variant', 'version',
                                    'on_top', 'cond', 'dimension', 'rt', 'acc'])
    df.to_csv('amir_2012.csv')

def behav_amir(**kwargs):
    df = pandas.read_csv('amir_2012.csv')
    df = df[df.version=='3d']
    df = df[~df.subjid.isin(['KA11','JJ'])]
    df = df[df.run!=15]
    df = df[~df.cond.isin([31,34])]
    df = df[df.acc==100]
    agg = stats.aggregate(df, groupby=['dimension', 'variant',
                          'version', 'subjid'])

    sns.factorplot(x='version',y='rt',hue='variant',col='dimension',
                    data=agg,kind='bar',col_wrap=3)
    sns.plt.show()

def acc_all(model_name='', layers=None, **kwargs):
    df = []
    deep = [d for d in base.DEEP if d != 'HMO']
    for model in base.SHALLOW + deep:
        spl = model.split()
        if len(spl) == 2:
            model_name, layer = spl
        else:
            model_name = model
            layer = None
        m = Geons(model_name=model_name, layers=layer, **kwargs)
        m.accuracy(plot=False)
        for rowno, acc in m.acc.iterrows():
            df.append([model_name] + acc.tolist()[1:])

    df = pandas.DataFrame(df, columns=['models', 'geon', 'accuracy'])

    sel = df[df.models.isin(['phow', 'CaffeNet'])]
    sel = sel.pivot_table(columns='models', index='accuracy',
                          aggfunc=len)
    print scipy.stats.fisher_exact(sel)

    df['kind'] = 'nap'
    dims = OrderedDict([('nap', [0,1])])

    orange = sns.color_palette('Set2', 8)[1]
    base.plot_all(df, dims, values='accuracy', colors=[orange],
                  pref='acc', ylim=[0,1], **kwargs)

def acc_all_layers(model_name=None, layers='all', **kwargs):
    colors = sns.color_palette('Set2')[1]

    fig, axes = sns.plt.subplots(len(base.DEEP), sharey=True, figsize=(2.5,4))
    for model, ax in zip(base.DEEP, axes):
        m = Geons(model_name=model, layers='all', **kwargs)
        m.accuracy(plot=False)
        sel = m.acc.groupby('layer').mean()

        ax.plot(range(len(sel)), np.array(sel.accuracy), lw=3, color=colors)
        ax.set_xlim([0, len(sel)-1])
        ax.set_xticklabels([])
        ax.set_title(model)
        sns.despine()

    sns.plt.ylim([0, 1])
    ax.set_yticks([0,.5,1])
    ax.set_yticklabels(['0','.5','1'])
    base.show(pref='acc', suffix='all_layers', **kwargs)

def run(**kwargs):
    getattr(Geons(**kwargs), kwargs['func'])()


class Compare(base.Compare):
    def __init__(self, *args):
        super(Compare, self).__init__(*args)

    def accuracy(self):
        return self.compare(pref='acc')

    def misclass(self, model_name=None, layers='all', **kwargs):
        colors = sns.color_palette('Set2')[1]

        fig, axes = sns.plt.subplots(len(base.DEEP), sharey=True, figsize=(2.5,4))
        for model, ax in zip(base.DEEP, axes):
            m = Geons(model_name=model, layers=None, **kwargs)
            m.accuracy(plot=False)
            print 'misclassified geons by %s:' % model
            print m.acc[m.acc.accuracy==False][['fno','dimension']]


def report(**kwargs):
    html = kwargs['html']
    kwargs['subset'] = '3d'
    html.writeh('Geons', h='h1')

    html.writeh('Accuracy', h='h2')

    kwargs['layers'] = 'all'
    kwargs['task'] = 'run'
    kwargs['func'] = 'accuracy'
    myexp = Geons(**kwargs)
    for depth, model_name in myexp.models:
        myexp.set_model(model_name)
        if depth != 'shallow':
            myexp.accuracy()

    kwargs['layers'] = 'output'
    kwargs['task'] = 'compare'
    myexp = Geons(**kwargs)
    Compare(myexp).accuracy()
