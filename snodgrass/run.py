import os, sys, glob
from collections import OrderedDict

import numpy as np
import pandas
from nltk.corpus import wordnet as wn
import seaborn as sns
import statsmodels.formula.api as smf

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import models, stats, plot

ORDER = ['color', 'gray', 'sil']
import base

class Snodgrass(base.Base):

    def __init__(self, *args, **kwargs):
        super(Snodgrass, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.dims = OrderedDict([('shape', np.repeat(range(6),6))])
        self.colors = OrderedDict([('shape', base.COLORS[1])])

    def pred_acc(self, compute_acc=True):
        if compute_acc:
            preds = self.predict()
        imagenet_labels = self.synsets_from_txt('snodgrass/synset_words.txt')
        dataset_labels = self.synsets_from_csv(os.path.join('snodgrass', self.exp + '.csv'))
        all_hyps = lambda s:s.hyponyms()

        df = pandas.DataFrame.from_dict(dataset_labels)
        df['imgid'] = ''
        df['imdnames'] = ''
        df['kind'] = 'unknown'
        df['accuracy'] = np.nan
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
                for p in preds[no]:
                    psyn = wn._synset_from_pos_and_offset(p['synset'][0],
                                                          int(p['synset'][1:]))
                    # check if the prediction is exact
                    # or at least more specific than the correct resp
                    if psyn in hypos:
                        df.loc[no, 'accuracy'] = True
                        break
                else:
                    if df.loc[no, 'kind'] != 'unknown':
                        df.loc[no, 'accuracy'] = False
        return df

    def acc_single(self):
        df = self.pred_acc()
        acc_exact = df[df.kind=='exact'].accuracy.mean()
        print 'Exact match: {:.2f}'.format(acc_exact)
        print 'Exact match or more specific: {:.2f}'.format(df.accuracy.mean())
        return df

    def accuracy(self):
        dfs = []
        for subset in ORDER:
            self.set_subset(subset)
            df = self.acc_single()
            df['model'] = self.model_name
            df['dataset'] = subset
            dfs.append(df)
        df = pandas.concat(dfs, ignore_index=True)
        return df

    def corr(self):
        self.set_subset('sil')
        df = self.acc_single()
        df['dataset'] = 'sil'
        human = pandas.read_csv('snodgrass/sil_human_acc.csv',header=None)
        df['human_accuracy'] = np.nan
        n = len(df.dataset.unique())
        df.loc[:,'human_accuracy'] = human[0].tolist() * n
        df['human_accuracy'] /= 100.
        df = df[df.kind!='unknown']
        df.accuracy = df.accuracy.astype(int)
        sns.set_palette(sns.color_palette('Set2')[1:])

        self._corr(df, 'all')
        self._corr(df[df.kind=='exact'], 'exact')

    def _corr(self, sel, suffix):
        formula = 'accuracy ~ human_accuracy'
        logreg = smf.logit(formula=formula, data=sel).fit()
        summ = logreg.summary()
        if self.html is None:
            print summ
        else:
            summ = summ.as_html().replace('class="simpletable"',
                                          'class="simpletable table"')

        sel = sel.rename(columns={'human_accuracy': 'human accuracy'})

        sns.lmplot('human accuracy', 'accuracy', data=sel,
                    y_jitter=.05, logistic=True, truncate=True)

        bins = np.digitize(sel['human accuracy'], np.arange(0,1,.1))
        bins[bins==11] = 10
        count = sel.accuracy.groupby(bins).count()
        mean = sel.accuracy.groupby(bins).mean()
        sns.plt.scatter(.1*mean.index-.05, mean, s=10*count, c='.15',
                        linewidths=0, alpha=.8)
        sns.plt.title(models.NICE_NAMES[self.model_name])
        sns.plt.xlim([-.1, 1.1])
        sns.plt.ylim([-.1, 1.1])

        self.show(pref='corr_sil', suffix=self.model_name + '_' + suffix,
                  caption=suffix + summ)

    def behav(self):
        self.model_name = 'behav'
        df = [('color', .903, .169),
              ('gray', .892, .172),
            #   ('line drawing', .882, .171),
              ('silhouette', .6470, .3976)]
        df = pandas.DataFrame(df, columns=['dataset', 'accuracy', 'stdev'])
        n = 260
        ci = df.stdev * 1.96 / np.sqrt(n)
        df['ci_low'] = df.accuracy - ci
        df['ci_high'] = df.accuracy + ci

        sns.factorplot('dataset', 'accuracy', data=df,
                        kind='bar', color=self.colors['shape'])
        plot.plot_ci(df, what=['Rectangle'])
        self.show(pref='acc')

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


class Compare(base.Compare):
    def __init__(self, *args):
        super(Compare, self).__init__(*args)

    def accuracy(self):
        df = self._acc()
        df = df[df.kind!='unknown']
        df.dataset[df.dataset=='sil'] = 'silhouette'

        sns.factorplot('dataset', 'accuracy', 'model', data=df,
                        kind='bar', color=self.myexp.colors['shape'])
        sns.plt.ylim([0,1])
        self.show(pref='acc')
        return df

    def _acc(self):
        dfs = []
        for depth, model_name in self.myexp.models:
            if depth == 'deep':
                self.myexp.set_model(model_name)
                df = self.myexp.accuracy()
                dfs.append(df)
        df = pandas.concat(dfs, ignore_index=True)
        return df


def report(**kwargs):
    html = kwargs['html']
    html.writeh('Snodgrass', h='h1')

    html.writeh('Accuracy', h='h2')

    html.writeh('Behavioral', h='h3')
    kwargs['layers'] = 'probs'
    kwargs['task'] = 'run'
    kwargs['func'] = 'behav'
    myexp = Snodgrass(**kwargs)
    myexp.behav()

    html.writeh('Models', h='h3')
    kwargs['layers'] = 'preds'
    kwargs['task'] = 'compare'
    kwargs['func'] = 'accuracy'
    myexp = Snodgrass(**kwargs)
    df = Compare(myexp).accuracy()

    html.writeh('Correlation', h='h2')
    kwargs['layers'] = 'probs'
    kwargs['task'] = 'run'
    kwargs['func'] = 'corr'
    myexp = Snodgrass(**kwargs)
    for depth, model_name in myexp.models:
        if depth == 'deep':
            myexp.set_model(model_name)
            myexp.corr()
