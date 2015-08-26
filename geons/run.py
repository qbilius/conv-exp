import os, sys
from collections import OrderedDict

import numpy as np
import scipy.stats
import pandas
import seaborn as sns

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import models, stats

import base
from base import Base

base.DEEP = ['CaffeNet', 'VGG-19', 'GoogleNet']

class Geons(Base):

    def __init__(self, *args, **kwargs):
        super(Geons, self).__init__(*args, **kwargs)
        self.kwargs = kwargs

    def get_dims(self):
        with open('dimensions.csv', 'rb') as f:
            lines = f.readlines()
        dims = dict([l.strip('\n').split(',') for l in lines])
        return dims

    @base._check_force('nap')
    def dissimilarity(self):
        self.classify()
        df = []
        dims = self.get_dims()
        for layer, resps in self.resps.items():
            for g,i in enumerate(range(0, len(resps), 3)):
                dis = super(Base, self).dissimilarity(resps[i:i+3],
                                                      kind='corr')
                n = os.path.basename(self.ims[i])[:2]
                df.append([layer, g, int(n), dims[n], 'non-accidental', dis[1,0]])
                df.append([layer, g, int(n), dims[n], 'metric', dis[1,2]])

        self.nap = pandas.DataFrame(df,
                            columns=['layer', 'geon', 'fno', 'dimension', 'kind', 'dist'])
        self.save('nap')

    def accuracy(self, plot=True):
        self.dissimilarity()
        dff = self.nap.pivot_table(values='dist', index=['layer', 'geon', 'fno', 'dimension'], columns='kind')

        dff = pandas.DataFrame(dff.apply(np.argmax, axis=1), columns=['accuracy'])
        dff = dff.reset_index()
        dff.loc[dff.accuracy=='metric', 'accuracy'] = False
        dff.loc[dff.accuracy=='non-accidental', 'accuracy'] = True
        dff.accuracy = dff.accuracy.astype(bool)
        self.acc = dff
        # import pdb; pdb.set_trace()
        # self.acc = stats.aggregate(dff, values='accuracy', rows='layer')
        # acc = stats.accuracy(dff, values='accuracy', rows='layer',
        #                 correct='non-accidental', incorrect='metric')
        # self.acc = pandas.DataFrame(acc).reset_index()
        if plot:
            self.plot_acc()

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


# def compare_lin(**kwargs):
#     lin = base.compare(Geons, 'accuracy', 'acc', **kwargs)
#     g = sns.factorplot('model', 'accuracy', data=lin, kind='bar',
#                       color='indianred')
#     g.axes.flat[0].axhline(1/2., ls='--', c='.2')
#     g.axes.flat[0].set_ylim([0,1])
#     base.show(pref='acc', **kwargs)

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
        # if layer is not None:
        #     m.acc = m.acc[m.acc.layer == layer]

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

    # base.save(df, pref='corr', suffix='CaffeNet', **kwargs)
    # gray = sns.color_palette('Set2', 8)[-1]
    # orange = sns.color_palette('Set2', 8)[1]
    # palette = [gray] * len(shallow) + [orange] * len(deep)
    # g = sns.factorplot('models', 'accuracy', data=df,
    #                     kind='bar', palette=palette)


    # base.set_vertical_labels(g)

    # # chances = [1./len(np.unique(val)) for val in Stefania().dims.values()]
    # # for ax, chance in zip(g.axes.flat, chances):
    # sns.plt.axhline(.5, ls='--', c='.2')



    # base.show(pref='corr', **kwargs)

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

def misclass(model_name=None, layers='all', **kwargs):
    colors = sns.color_palette('Set2')[1]

    fig, axes = sns.plt.subplots(len(base.DEEP), sharey=True, figsize=(2.5,4))
    for model, ax in zip(base.DEEP, axes):
        m = Geons(model_name=model, layers=None, **kwargs)
        m.accuracy(plot=False)
        print 'misclassified geons by %s:' % model
        print m.acc[m.acc.accuracy==False][['fno','dimension']]

def run(**kwargs):
    getattr(Geons(**kwargs), kwargs['func'])()
