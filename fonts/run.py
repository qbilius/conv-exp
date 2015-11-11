import sys, os, glob
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import scipy.io
import pandas
import seaborn as sns
import sklearn.cluster, sklearn.metrics

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import models, stats

import base


class Fonts(base.Base):

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.dims = OrderedDict([('shape', np.repeat(range(6),6))])
        self.colors = OrderedDict([('shape', base.COLORS[1])])
        self.skip_hmo = True
        super(Fonts, self).__init__(*args, **kwargs)

    def behav(self):
        dfiles = glob.glob('fonts/multipleArrangements/results/fonts_*_2015*.mat')
        n = self.dims['shape'].size
        inds = np.triu_indices(n, k=1)
        data = np.ones((len(dfiles), n, n)) * np.nan

        for i,d in enumerate(dfiles):
            data[i][inds] = scipy.io.loadmat(d)['estimate_dissimMat_ltv']
            data[i].T[inds] = data[i][inds]
        behav = OrderedDict([('shape', data)])
        self.save(behav, 'behav')

    def cluster_behav(self):
        """
        TO-DO: make a resampled version?
        """
        df = pandas.read_csv('fonts/data/clust_fonts_behav.csv')
        sc = []
        for subjid in df.subjid.unique():
            sel = df[df.subjid==subjid]
            ari = sklearn.metrics.adjusted_rand_score(sel.label, sel.user_label)
            sc.append([subjid, ari])
        sc = pandas.DataFrame(sc, columns=['subjid', 'dissimilarity'])
        return sc


class Compare(base.Compare):
    def __init__(self, *args):
        super(Compare, self).__init__(*args)

    def cluster(self):
        # self.myexp.savedata = False
        # force = myexp.force
        # self.myexp.force = True
        self.myexp.bootstrap = True
        return self.compare(pref='clust')


def report(**kwargs):

    html = kwargs['html']
    kwargs['bootstrap'] = True

    html.writeh('Fonts', h='h1')

    html.writeh('Clustering', h='h2')

    kwargs['layers'] = 'all'
    kwargs['task'] = 'run'
    kwargs['func'] = 'cluster'
    myexp = Fonts(**kwargs)
    for depth, model_name in myexp.models:
        if depth != 'shallow':
            myexp.set_model(model_name)
            myexp.cluster()

    kwargs['layers'] = 'output'
    kwargs['task'] = 'compare'
    myexp = Fonts(**kwargs)
    Compare(myexp).cluster()

    html.writeh('Correlation', h='h2')

    kwargs['layers'] = 'all'
    kwargs['task'] = 'run'
    kwargs['func'] = 'corr'
    myexp = Fonts(**kwargs)
    for depth, model_name in myexp.models:
        if depth != 'shallow':
            myexp.set_model(model_name)
            myexp.corr()

    kwargs['layers'] = 'output'
    kwargs['task'] = 'compare'
    myexp = Fonts(**kwargs)
    Compare(myexp).corr()
