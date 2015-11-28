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
        kwargs['skip_hmo'] = True
        super(Fonts, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.dims = OrderedDict([('px', np.repeat(range(6),6)),
                                 ('shape', np.repeat(range(6),6))])
        # self.colors = OrderedDict([('shape', base.COLORS[1])])
        self.colors = OrderedDict([('px', base.COLORS[0]),
                                   ('shape', base.COLORS[1])])

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

    def mds(self):
        p = sns.color_palette('Set2', 8)
        colors = [p[1], p[5], p[6], p[2], p[3], p[7]]
        icons = []
        ims = sorted(glob.glob('fonts/img/alpha/*.png'))
        for i,c in enumerate(self.dims['shape']):
            icon = models.load_image(ims[i], keep_alpha=True)
            # import pdb; pdb.set_trace()

            icon[:,:,:3][icon[:,:,3]>0] = colors[c][0], colors[c][1], colors[c][2]
            # for j in range(3):
            #     icon[:,:,j][icon[:,:,j]<.5] = colors[c][j]
            icons.append(icon)
        super(Fonts, self).mds(icons=icons, seed=0, zoom=.2)

    def gen_letters(self):
        import subprocess, string
        fonts = sorted(glob.glob('fonts/data/fonts/*.ttf'))
        sizes = {'arcadian': 500, 'atlantean': 400, 'dovahzul': 400,
                 'futurama': 500, 'hymmnos': 400, 'ulog': 1000}
        cmd = ('convert -font {} -pointsize {} -background {} '
               'label:{} -trim '
               '-gravity center -extent 512x512 -resize 256x256 {}')
        for bckg in ['white', 'none']:
            for font in fonts:
                fontname = os.path.basename(font).split('.')[0]
                for letter in string.ascii_lowercase[:6]:
                    name = fontname + '_' + letter + '.png'
                    if bckg == 'white':
                        newname = os.path.join('fonts', 'img', name)
                    else:
                        newname = os.path.join('fonts', 'img', 'alpha', name)
                    subprocess.call(cmd.format(font, sizes[fontname], bckg, letter, newname).split())


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
    html.writeh('Fonts', h='h1')

    # html.writeh('Clustering', h='h2')
    #
    # kwargs['layers'] = 'all'
    # kwargs['task'] = 'run'
    # kwargs['func'] = 'cluster'
    # myexp = Fonts(**kwargs)
    # for depth, model_name in myexp.models:
    #     if depth != 'shallow':
    #         myexp.set_model(model_name)
    #         myexp.cluster()
    #
    # kwargs['layers'] = 'output'
    # kwargs['task'] = 'compare'
    # myexp = Fonts(**kwargs)
    # Compare(myexp).cluster()

    html.writeh('MDS', h='h2')
    kwargs['layers'] = 'output'
    kwargs['task'] = 'run'
    kwargs['func'] = 'mds'
    myexp = Fonts(**kwargs)
    for name in ['shape', 'googlenet']:
        myexp.set_model(name)
        myexp.mds()

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
    kwargs['force'] = False
    kwargs['forceresps'] = False
    myexp = Fonts(**kwargs)
    Compare(myexp).corr()
