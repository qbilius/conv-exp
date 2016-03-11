from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, urllib
from collections import OrderedDict, Counter

import numpy as np
import scipy.stats
import pandas
import seaborn as sns

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import models, stats

import base


class Geons(base.Base):

    def __init__(self, *args, **kwargs):
        kwargs['skip_hmo'] = True
        super(Geons, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.dims = OrderedDict([('nap', [0,1])])
        self.colors = OrderedDict([('nap', base.COLORS[1])])

    def get_images(self):
        url_root = 'http://geon.usc.edu/~ori/'
        html = urllib.urlopen(url_root + 'VogelsShaded124.html').read()
        spl = html.split('src="')
        path = os.path.join('geons', 'img')
        if not os.path.isdir(path): os.makedirs(path)
        print('Downloading images...')
        for sp in spl[1:]:
            url = sp.split('"')[0]
            urllib.urlretrieve(url_root + url, os.path.join(path, os.path.basename(url)))

    @base.get_data('nap')
    def dissimilarity(self):
        resps = self.classify()
        df = []
        dims = self.get_dims()
        for layer, resps in resps.items():
            for g,i in enumerate(range(0, len(resps), 3)):
                dis = models.dissimilarity(resps[i:i+3],
                                                      kind='correlation')
                n = int(os.path.basename(self.ims[i]).split('os')[0])
                df.append([layer, g, n, dims[n], 'non-accidental', dis[1,0]])
                df.append([layer, g, n, dims[n], 'metric', dis[1,2]])

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

    def errors(self):
        nap = self.dissimilarity()
        acc = nap.pivot_table(values='dist', index=['layer', 'geon', 'fno', 'dimension'], columns='kind').reset_index()
        acc['accuracy'] = acc['non-accidental'] > acc['metric']
        err = acc[acc.accuracy==False].dimension
        count = [(models.NICE_NAMES[self.model_name].lower(),d,sum(err==d)) for d in np.unique(acc.dimension)]
        cdf = pandas.DataFrame(count, columns=['models', 'dimension', 'count'])
        return cdf

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

    def get_dims(self):
        with open('geons/data/dimensions.csv', 'rb') as f:
            lines = f.readlines()
        dims = {}
        for l in lines:
            spl = l.strip('\n').split(',')
            dims[int(spl[0])] = spl[1]
        return dims

    def remake_amir2012(self):
        data = scipy.io.loadmat('amir2012.mat')
        meta = data['Results'][0,0]
        acc = meta['CRmat']
        ims = data['ImageNames']
        df = []
        variants = {0: 'metric', 1: 'non-accidental'}
        vobs = {0: 'variant', 1: 'base'}
        versions = {0: '3d', 1: '2d'}
        dims = self.get_dims()

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

    def behav_amir(self):
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


class Compare(base.Compare):
    def __init__(self, *args):
        super(Compare, self).__init__(*args)

    def accuracy(self):
        return self.compare(pref='acc', ylim=[0,1])

    def errors(self):
        colors = sns.color_palette('Set2')[1]
        df = []
        for depth, model in self.myexp.models:
            self.myexp.set_model(model)
            e = self.myexp.errors()
            for i,r in e.iterrows():
                df.append([depth] + r.values.tolist())
        df = pandas.DataFrame(df, columns=['depth']+e.columns.values.tolist())
        sns.factorplot(x='dimension', y='count', data=df, hue='depth', kind='bar')
        self.show(pref='errors', suffix='all')


def report(**kwargs):
    html = kwargs['html']
    # kwargs['subset'] = '3d'
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
    kwargs['force'] = False
    kwargs['forcemodels'] = False
    myexp = Geons(**kwargs)
    Compare(myexp).accuracy()
    Compare(myexp).errors()
