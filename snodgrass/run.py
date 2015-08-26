import os, sys, glob

import numpy as np
import pandas
from nltk.corpus import wordnet as wn
import seaborn as sns
import statsmodels.formula.api as smf

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import stats

import base
from base import Base
base.DEEP = ['CaffeNet', 'VGG-19', 'GoogleNet']
ORDER = ['color', 'gray', 'sil']

class Snodgrass(Base):

    def __init__(self, *args, **kwargs):
        super(Snodgrass, self).__init__(*args, **kwargs)
        self.kwargs = kwargs

    def accuracy(self):
        # dataset_labels = self.synsets_from_csv(self.dataset + '.csv')
        # self.predict()
        # df = self.pred_acc(dataset_labels)
        df = self.pred_acc()
        # print df[['id','imgid','kind','accuracy']]
        # sys.exit()
        acc_exact = df[df.kind=='exact'].accuracy.mean()
        # acc_exact = acc.sum() / float(df.accuracy.count())
        print 'Exact match: {:.2f}'.format(acc_exact)
        print 'Exact match or more specific: {:.2f}'.format(df.accuracy.mean())
        return df

    def create_stim(self):
        suffix = self.order

        gen2nd = GenSecondOrder()
        for fn in sorted(glob.glob('shape/snodgrass_color_orig/*.jpg')):
            fname = os.path.basename(fn)
            dim = max(imread(fn).shape)
            if dim != 281: print fname, dim

            newname = fname.split('.')[0] + '.png'

            # square color images
            cname = os.path.join('datasets', 'snodgrass', suffix[0], newname)
            # import pdb; pdb.set_trace()

            subprocess.call('convert {0} -alpha set -channel RGBA -fuzz 10% '
                            '-fill none -floodfill +0+0 white '
                            '-gravity center -background none '
                            '-extent {2}x{2} {1}'.format(fn, cname, dim).split())

            # grayscaled
            newfname = os.path.join('datasets', 'snodgrass', suffix[1], newname)
            subprocess.call('convert {} -channel RGBA -matte -colorspace '
                            'gray {}'.format(cname, newfname).split())

            # line drawings
            newfname = os.path.join('datasets', 'snodgrass', suffix[2], newname)
            subprocess.call('convert {} -negate -separate -lat 5x5+5% -negate '
                            '-evaluate-sequence add {}'.format(cname, newfname).split())

            # silhouettes
            newfname = os.path.join('datasets', 'snodgrass', suffix[3], newname)
            subprocess.call('convert {} -blur 1x1 -alpha extract '
                            '-negate {}'.format(cname, newfname).split())

            # 2nd order edges
            im = imread(newfname, flatten=True)
            mask = np.zeros(im.shape).astype(bool)
            mask[im==0] = True
            newfname = os.path.join('datasets', 'snodgrass', suffix[4], newname)
            gen2nd.gen(mask=im, savename=newfname)


def acc(model_name, subset=None, plot=True, **kwargs):
    dff = []

    for subset in ORDER:
        m = Snodgrass(model_name=model_name, subset=subset, **kwargs)
        df = m.accuracy()
        df['model'] = model_name
        df['dataset'] = subset
        dff.append(df)
        # df_exact = df[df.id==df.imgid]  # has that exact synset
        # dff_exact.append(df_exact)
        # acc = df_exact.acc.sum() / float(df.acc.count())
        # acc_exact = df_exact.acc.mean()
        # acc_all = df.acc.mean()

        # dff.append([dataset, 'exact', acc])
        # dff.append([dataset, 'exact or more specific', acc_all])
        # dff_exact.append([dataset, acc_exact])

    dff = pandas.concat(dff, ignore_index=True)

    human = pandas.read_csv('sil_human_acc.csv',header=None)[0].tolist()
    dff['human_accuracy'] = np.nan
    n = len(dff.dataset.unique())
    dff.loc[:,'human_accuracy'] = human * n

    # print '=== Easy for humans, hard for model ==='
    # print dff[dff.dataset=='color'][dff.accuracy==False][dff.human_accuracy>50][['names', 'imgnames']]
    # return dff
    if plot:
        dff = dff[dff.kind!='unknown']
        print '# of exact synsets:', dff[dff.kind=='exact'].accuracy.count()/n
        print '# of exact or more specific synsets:', dff.accuracy.count()/n
        _acc(dff, 'all', **kwargs)
        _acc(dff[dff.kind=='exact'], 'exact', **kwargs)

    return dff

def _acc(dff, suffix, **kwargs):
    sel = dff[dff.dataset=='sil']
    sel.accuracy = sel.accuracy.astype(int)
    formula = 'accuracy ~ human_accuracy'
    logreg = smf.logit(formula=formula, data=sel).fit()
    print logreg.summary()

    sel = sel.rename(columns={'human_accuracy': 'human accuracy'})

    sns.lmplot('human accuracy', 'accuracy', data=sel,
                y_jitter=.05, logistic=True)

    bins = np.digitize(sel['human accuracy'], range(0,100,10))
    bins[bins==11] = 10
    count = sel.accuracy.groupby(bins).count()
    mean = sel.accuracy.groupby(bins).mean()
    sns.plt.scatter(10*mean.index-5, mean, s=10*count, c='.15',#'#66c2a5',
     linewidths=0, alpha=.8)
    base.show(pref='log', suffix='sil_'+suffix, **kwargs)

    orange = sns.color_palette('Set2')[1]
    sns.factorplot('dataset', 'accuracy', data=dff,
                    kind='bar', color=orange)
    sns.plt.ylim([0,1])
    base.show(pref='acc', suffix=suffix, **kwargs)

def acc_all(model_name=None, **kwargs):
    df = pandas.concat([acc(model, plot=False, **kwargs) for model in base.DEEP], ignore_index=True)
    # df.dataset[df.dataset=='line'] = 'line drawing'
    df.dataset[df.dataset=='sil'] = 'silhouette'

    orange = sns.color_palette('Set2')[1]
    sns.factorplot('dataset', 'accuracy', 'model', data=df,
                    kind='bar', color=orange)
    sns.plt.ylim([0,1])
    base.show(pref='acc_all', suffix='all', **kwargs)

def behav(**kwargs):
    df = [('color', .903, .169),
          ('gray', .892, .172),
        #   ('line drawing', .882, .171),
          ('silhouette', .6470, .3976)]
    df = pandas.DataFrame(df, columns=['dataset', 'accuracy', 'stdev'])
    n = 260
    ci = df.stdev * 1.96 / np.sqrt(n)
    df['ci_low'] = df.accuracy - ci
    df['ci_high'] = df.accuracy + ci

    orange = sns.color_palette('Set2')[1]
    sns.factorplot('dataset', 'accuracy', data=df,
                    kind='bar', color=orange)
    base.plot_ci(df, what=['Rectangle'])
    base.show(pref='acc', suffix='behav', **kwargs)

def run(**kwargs):
    getattr(Snodgrass(**kwargs), kwargs['func'])()
