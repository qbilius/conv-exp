from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os, glob, subprocess
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import scipy.io
import pandas
import seaborn as sns
import sklearn.cluster, sklearn.metrics

from psychopy_ext import stats, utils

import base


class HOP2008(base.Base):

    def __init__(self, *args, **kwargs):
        kwargs['skip_hmo'] = False
        super(HOP2008, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.dims = OrderedDict([
                        ('px', np.array([0,0,0,1,1,1,2,2,2])),
                        ('shape', np.array([0,1,2,1,2,0,2,0,1]))])
        self.colors = OrderedDict([('px', base.COLORS[0]),
                                   ('shape', base.COLORS[1])])

    def get_images(self):
        self._gen_alpha()

    def mds(self):
        path = os.path.join('hop2008', 'img', 'alpha', '*.*')
        icons = sorted(glob.glob(path))
        super(HOP2008, self).mds(icons=icons, seed=3)  # to match behav


    # def plot_lin(self, subplots=False):
    #     super(HOP2008, self).plot_lin(subplots=False)

    # def corr_mod(self):
    #     self.dissimilarity()
    #     human = pickle.load(open('dis_hop2008_human.pkl', 'rb'))
    #     df = []
    #     for label, data in human.items():
    #         d = self.dis[self.dis.keys()[-1]]
    #         inds = np.triu_indices(d.shape[0], k=1)
    #         corr = np.corrcoef(data[inds], d[inds])[0,1]
    #         df.append([label, corr])
    #     df = pandas.DataFrame(df, columns=['layer', 'correlation'])
    #     sns.factorplot('layer', 'correlation', data=df,
    #                   color='steelblue', kind='bar')
    #     self.show('corr_mod')


    # def plot_linear_clf(self):
    #     xlabel = '%s layers' % self.model_name
    #     self.lin = self.lin.rename(columns={'layer': xlabel})
    #     if self.model_name == 'GoogleNet':
    #         g = sns.factorplot(xlabel, 'accuracy', 'kind', data=self.lin,
    #                           kind='point', markers='None', legend=False, ci=0)
    #         g.axes.flat[0].set_xticklabels([])

    #         import matplotlib.lines as mlines
    #         colors = sns.color_palette('Set2', 8)[:len(self.dims)]
    #         handles = []
    #         for mname, color in zip(self.dims.keys(), colors):
    #             patch = mlines.Line2D([], [], color=color, label=mname)
    #             handles.append(patch)
    #         g.axes.flat[0].legend(handles=handles, loc='best')
    #     else:
    #         g = sns.factorplot(xlabel, 'accuracy', 'kind', data=self.lin,
    #                           kind='point')

    #     g.axes.flat[0].axhline(1/3., ls='--', c='.2')
    #     self.show(pref='lin')

    def avg_hop2008(self, dis, plot=True):
        # dis = self.dissimilarity()

        df = []
        for layer, dis in dis.items():

            df.extend(self._avg(dis, layer, 'px'))
            df.extend(self._avg(dis, layer, 'shape'))

            other = dis.copy()
            for sh in np.unique(self.dims['px']):
                ss = self.dims['px'] == sh
                other.T[ss].T[ss] = np.nan

            for sh in np.unique(self.dims['shape']):
                ss = self.dims['shape'] == sh
                other.T[ss].T[ss] = np.nan

            inds = range(len(other))
            n = 0
            for si, s1 in enumerate(inds):
                for s2 in inds[si+1:]:
                    if not np.isnan(other[s1,s2]):
                        df.append([layer, 'other', n, s1, s2, other[s1,s2]])
                        n += 1

        df = pandas.DataFrame(df, columns=['layer', 'kind', 'n', 'i', 'j', 'dissimilarity'])
        if plot:
            df = stats.factorize(df, order={'kind': ['px', 'shape', 'other']})
            # df = df[df.feature != 'other']
            # agg = stats.aggregate(df, values='dissimilarity', rows='layer',
            #                  cols='feature', yerr='n')
            agg = df.pivot_table(index='n', columns=['layer', 'kind'],
                           values='dissimilarity')
            sns.factorplot('layer', 'dissimilarity', 'kind', data=df,
                            kind='bar')
            self.show(pref='avg')
        return df

    def _avg(self, dis, layer, name):
        df = []
        n = 0
        inds = np.arange(len(self.dims[name]))
        for sh in np.unique(self.dims[name]):
            sel = inds[self.dims[name]==sh]
            for si, s1 in enumerate(sel):
                for s2 in sel[si+1:]:
                    df.append([layer, name, n, s1, s2, dis[s1,s2]])
                    n += 1
        return df

    def dis_group_diff(self, plot=True):
        group = self.dis_group(plot=False)
        agg = group.pivot_table(index=['layer','n'], columns='kind',
                             values='similarity').reset_index()
        agg['diff'] = agg['shape'] - agg['px']
        if self.bootstrap:
            dfs = []
            for layer in agg.layer.unique():
                sel = agg[agg.layer==layer]['diff']
                pct = stats.bootstrap_resample(sel, ci=None, func=np.mean)
                d = OrderedDict([('kind', ['diff'] * len(pct)),
                                 ('layer', [layer]*len(pct)),
                                 ('preference', sel.mean()),
                                 ('iter', range(len(pct))),
                                 ('bootstrap', pct)])
                dfs.append(pandas.DataFrame.from_dict(d))
            df = pandas.concat(dfs)

        else:
            df = agg.groupby('layer').mean().reset_index()
            df = df.rename(columns={'diff':'preference'})
            df['kind'] = 'diff'
            df['iter'] = 0
            df['bootstrap'] = np.nan
            del df['px']
            del df['shape']
        return df

    def plot_dis_group_diff(self, subplots=False):
        xlabel = '%s layer' % self.model_name
        self.diff = self.diff.rename(columns={'layer': xlabel})
        orange = sns.color_palette('Set2', 8)[1]
        self.plot_single_model(self.diff, subplots=subplots, colors=orange)
        sns.plt.axhline(0, ls='--', c='.15')

        sns.plt.ylim([-.2, .8])
        self.show('dis_group_diff')

    def dis_group(self, plot=True):
        dis = self.dissimilarity()
        df = self.avg_hop2008(dis, plot=False)

        df = df[df.kind != 'other'][['layer', 'kind', 'n', 'dissimilarity']]
        df['similarity'] = 1 - df.dissimilarity
        group = df.copy()
        del group['dissimilarity']
        print(group)

        if self.task == 'run' and plot:
            self.plot_single(group, 'dis_group')
        return group

    def _gen_alpha(self):
        path = 'hop2008/img/alpha'
        if not os.path.isdir(path): os.makedirs(path)
        for f in sorted(glob.glob('hop2008/img/*.tif')):
            fname = os.path.basename(f)
            newname = fname.split('.')[0] + '.png'
            newname = os.path.join(path, newname)
            # and now some ridiculousness just because ImageMagick can't make
            # alpha channel for no reason
            alphaname = fname.split('.')[0] + '_alpha.png'
            alphaname = os.path.join(path, alphaname)

            subprocess.call('convert {} -alpha set -channel RGBA '
                            '-fuzz 10% -fill none '
                            '-floodfill +0+0 rgba(100,100,100,0) '
                            '{}'.format(f, newname).split())
            subprocess.call('convert {} -alpha set -channel RGBA '
                            '-fuzz 10% -fill none '
                            '-floodfill +0+0 rgb(100,100,100) '
                            '-alpha extract {}'.format(f, alphaname).split())
            im = utils.load_image(newname)
            alpha = utils.load_image(alphaname)
            scipy.misc.imsave(newname, np.dstack([im,im,im,alpha]))
            os.remove(alphaname)

def remake_hop2008(**kwargs):
    data = scipy.io.loadmat('hop2008_behav.mat')
    data = np.array(list(data['behavioralDiff8'][0]))
    # reorder such that all recta things are last, not first
    a11 = data[:,:3,:3]
    a12 = data[:,:3,3:]
    a21 = data[:,3:,:3]
    a22 = data[:,3:,3:]
    b1 = np.concatenate([a22,a21], axis=2)
    b2 = np.concatenate([a12,a11], axis=2)
    b = np.concatenate([b1,b2], axis=1)
    pickle.dump(b, open('dis_hop2008_behav.pkl', 'wb'))

def ceil_rel(**kwargs):
    data = pickle.load(open('dis_hop2008_behav.pkl', 'rb'))
    inds = np.triu_indices(data.shape[1], k=1)
    df = np.array([d[inds] for d in data])
    zmn = np.mean(scipy.stats.zscore(df, axis=1), axis=0)
    ceil = np.mean([np.corrcoef(subj,zmn)[0,1] for subj in df])
    rng = np.arange(df.shape[0])

    floor = []
    for s, subj in enumerate(df):
        mn = np.mean(df[rng!=s], axis=0)
        floor.append(np.corrcoef(subj,mn)[0,1])
    floor = np.mean(floor)
    return floor, ceil


class Compare(base.Compare):
    def __init__(self, *args):
        super(Compare, self).__init__(*args)

    def dis_group(self):
        return self.compare(pref='dis_group')

    def dis_group_diff(self):
        return self.compare(pref='dis_group_diff', ylim=[-.4,.4])


def report(**kwargs):
    html = kwargs['html']

    html.writeh('HOP2008', h='h1')

    # html.writeh('Clustering', h='h2')
    #
    # kwargs['layers'] = 'all'
    # kwargs['task'] = 'run'
    # kwargs['func'] = 'dis_group'
    # myexp = HOP2008(**kwargs)
    # for depth, model_name in myexp.models:
    #     if depth != 'shallow':
    #         myexp.set_model(model_name)
    #         myexp.dis_group()
    #
    # kwargs['layers'] = 'output'
    # kwargs['task'] = 'compare'
    # kwargs['func'] = 'dis_group_diff'
    # myexp = HOP2008(**kwargs)
    # Compare(myexp).dis_group_diff()

    html.writeh('MDS', h='h2')
    kwargs['layers'] = 'output'
    kwargs['task'] = 'run'
    kwargs['func'] = 'mds'
    myexp = HOP2008(**kwargs)
    for name in ['px', 'shape', 'googlenet']:
        myexp.set_model(name)
        myexp.mds()

    html.writeh('Correlation', h='h2')

    kwargs['layers'] = 'all'
    kwargs['task'] = 'run'
    kwargs['func'] = 'corr'
    myexp = HOP2008(**kwargs)
    for depth, model_name in myexp.models:
        if depth != 'shallow':
            myexp.set_model(model_name)
            myexp.corr()

    kwargs['layers'] = 'output'
    kwargs['task'] = 'compare'
    kwargs['func'] = 'corr'
    kwargs['force'] = False
    kwargs['forcemodels'] = False
    myexp = HOP2008(**kwargs)
    Compare(myexp).corr()
