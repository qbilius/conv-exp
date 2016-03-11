import sys, os, glob
import subprocess, string
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import scipy.io
import pandas
import seaborn as sns
import sklearn.cluster, sklearn.metrics

from psychopy_ext import utils

import base


class Fonts(base.Base):

    def __init__(self, *args, **kwargs):
        kwargs['skip_hmo'] = True
        super(Fonts, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.dims = OrderedDict([('px', np.repeat(range(6),6)),
                                 ('shape', np.repeat(range(6),6))])
        self.colors = OrderedDict([('px', base.COLORS[0]),
                                   ('shape', base.COLORS[1])])

    def get_images(self):
        """
        Font descriptions:
        - http://www.omniglot.com/conscripts/arcadian.php
        - http://www.omniglot.com/conscripts/atlantean.htm
        - http://www.omniglot.com/conscripts/dovahzul.htm
        - http://www.omniglot.com/conscripts/futurama.htm
        - http://www.omniglot.com/conscripts/hymmnos.htm
        - http://www.omniglot.com/conscripts/ulog.php
        """
        urls = [('arcadian', 'http://arcadia.island.free.fr/down/police.zip'),
                ('atlantean', 'http://www.disneyexperience.com/downloads/fonts/atlantisfont.zip'),
                ('dovahzul', 'http://dl.1001fonts.com/dragon-alphabet.zip'),
                ('futurama', 'ftp://slurmed.com/incoming/tfp/fonts/alien_alphabet_1_font.zip'),
                ('hymmnos', 'http://www.ffonts.net/Hymmnos.font.zip'),
                ('ulog', 'http://www.omniglot.com/fonts/ulog.zip')]
        path = 'fonts/data/fonts/'
        if not os.path.isdir(path): os.makedirs(path)
        fpaths = []
        for name, url in urls:
            fnames = super(Fonts, self).download_dataset(url, path=path, ext='.ttf')
            for old_path in fnames:
                if os.path.splitext(old_path)[1] == '.ttf':
                    new_path = os.path.join(path, name + '.ttf')
                    os.rename(old_path, new_path)
                    self._gen_letters(new_path)

    def _gen_letters(self, font_path):
        sizes = {'arcadian': 500, 'atlantean': 400, 'dovahzul': 400,
                 'futurama': 500, 'hymmnos': 400, 'ulog': 1000}
        cmd = ('convert -font {} -pointsize {} -background {} '
               'label:{} -trim '
               '-gravity center -extent 512x512 -resize 256x256 {}')
        path1 = os.path.join('fonts', 'img')
        path2 = os.path.join('fonts', 'img', 'alpha')
        if not os.path.isdir(path1): os.makedirs(path1)
        if not os.path.isdir(path2): os.makedirs(path2)
        for bckg in ['white', 'none']:
            fontname = os.path.basename(font_path).split('.')[0]
            for letter in string.ascii_lowercase[:6]:
                name = fontname + '_' + letter + '.png'
                if bckg == 'white':
                    newname = os.path.join(path1, name)
                else:
                    newname = os.path.join(path2, name)
                subprocess.call(cmd.format(font_path, sizes[fontname], bckg, letter, newname).split())

    def behav(self):
        dfiles = glob.glob('fonts/data/fonts_*_2015*.mat')
        n = self.dims['shape'].size
        inds = np.triu_indices(n, k=1)
        data = np.ones((len(dfiles), n, n)) * np.nan

        for i,d in enumerate(dfiles):
            data[i][inds] = scipy.io.loadmat(d)['estimate_dissimMat_ltv']
            data[i].T[inds] = data[i][inds]

        behav = OrderedDict([('shape', data)])
        path = os.path.join('fonts', 'data')
        name =  'dis_fonts_shape.pkl'
        name = os.path.join(path, name)
        if self.savedata:
            if not os.path.isdir(path): os.makedirs(path)
            pickle.dump(behav, open(name, 'wb'))
            base.msg('saved to', name)

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
            icon = utils.load_image(ims[i], keep_alpha=True)
            icon[:,:,:3][icon[:,:,3]>0] = colors[c][0], colors[c][1], colors[c][2]
            icons.append(icon)
        super(Fonts, self).mds(icons=icons, seed=0, zoom=.2)


class Compare(base.Compare):
    def __init__(self, *args):
        super(Compare, self).__init__(*args)

    def cluster(self):
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
    kwargs['forcemodels'] = False
    myexp = Fonts(**kwargs)
    Compare(myexp).corr()
