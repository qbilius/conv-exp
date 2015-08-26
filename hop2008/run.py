import sys, os, glob
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import scipy.io
import pandas
import seaborn as sns

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import models, stats

import base
from base import Base, Shape


class HOP2008(Shape):

    def __init__(self, *args, **kwargs):
        super(HOP2008, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.dims = OrderedDict([
                        ('px', np.array([0,0,0,1,1,1,2,2,2])),
                        ('shape', np.array([0,1,2,1,2,0,2,0,1]))])

    @base._check_force('dis')
    def dissimilarity(self):
        self.classify()
        self.dis = super(Base, self).dissimilarity(self.resps,
                                                   kind='euclidean')
        self.save('dis')

    def mds(self):
        self.dissimilarity()
        path = os.path.join('img', 'png', '*.*')
        ims = sorted(glob.glob(path))
        super(Base, self).mds(self.dis, ims, kind='metric', seed=11)  # just to match behavioral mds
        self.show(pref='mds')

    def corr(self):
        super(HOP2008, self).corr(['px', 'shape'], subplots=False,
                                  **self.kwargs)

    def plot_lin(self, subplots=False):
        super(HOP2008, self).plot_lin(subplots=False)

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

    def avg_hop2008(self, plot=True):
        self.dissimilarity()

        df = []
        for layer, dis in self.dis.items():

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

        self.dis_group(plot=False)

        diff = []
        f = lambda x, y: np.mean(x) - np.mean(y)
        for layer in self.group.layer.unique():
            sel = self.group[self.group.layer==layer]

            agg = sel.groupby(['kind', 'n']).mean()
            cis = stats.bootstrap_matrix(agg.loc['shape'],
                                         agg.loc['px'],
                                         func=f)
            d = agg.loc['shape'].mean() - agg.loc['px'].mean()
            diff.append([layer, d.values[0], cis[0], cis[1]])

        self.diff = pandas.DataFrame(diff,
                columns=['layer', 'preference for shape', 'ci_low', 'ci_high'])
        # self.diff['kind'] = 'kind'

        if plot:
            self.plot_dis_group_diff()


    def plot_dis_group_diff(self, subplots=False):
        xlabel = '%s layer' % self.model_name
        self.diff = self.diff.rename(columns={'layer': xlabel})
        orange = sns.color_palette('Set2', 8)[1]
        self.plot_single_model(self.diff, subplots=subplots, colors=orange)

        # g = sns.factorplot('layer', 'similarity', 'kind', data=diff,
        #                 kind='point', aspect=aspect)
        # diff['kind'] = 'kind'
        # base.plot_ci(diff)
        sns.plt.axhline(0, ls='--', c='.15')

        sns.plt.ylim([-.2, .8])
        self.show('dis_group_diff')

    def dis_group(self, plot=True, comp_diff=False):
        df = self.avg_hop2008(plot=False)
        df = df[df.kind != 'other'][['layer', 'kind', 'n', 'dissimilarity']]
        # df.pivot_table(index=['n', 'layer'], columns='feature',
        #                      values='dissimilarity')
        # diff = agg['shape'] - agg['px']
        group = []
        for layer, dis in self.dis.items():
            sel = df[df.layer==layer]
            inds = np.triu_indices(len(dis), k=1)
            norm = np.max(dis[inds])
            sel['similarity'] = 1 - sel.loc[:,'dissimilarity']/norm
            group.append(sel)

        self.group = pandas.concat(group, ignore_index=True)
        del self.group['dissimilarity']

        if plot:
            self.plot_dis_group()
        # self.diff.rename(columns={'dissimilarity':'similarity'}, inplace=True)

        #
        # return diff
        # self.plot_lin_diff(diff, None)
        # self.show(pref='diff')

    # @base.style_plot
    def plot_dis_group(self, subplots=False):#diff, values, aspect=1, **kwargs):
        xlabel = '%s layer' % self.model_name
        self.group = self.group.rename(columns={'layer': xlabel})
        self.plot_single_model(self.group, subplots=subplots, )
        sns.plt.ylim([0, .8])
        self.show('dis_group')
        # return g

    def gen_png(self):
        import subprocess

        for f in sorted(glob.glob('img/*.tif')):
            fname = os.path.basename(f)
            newname = fname.split('.')[0] + '.png'
            newname = os.path.join('img/png', newname)
            im = models.load_images(f)
            scipy.misc.imsave(newname, np.dstack([im,im,im]))
            subprocess.call('convert {} -alpha set -channel RGBA '
                            '-fuzz 40% -fill none '
                            '-floodfill +0+0 rgb(95,95,95) '
                            '{}'.format(newname, newname).split())
            print models.load_images(newname).shape
            sys.exit()

    def place_on_bckg(self):
        import subprocess
        bckgs = sorted(glob.glob('img/bckg/*.jpg'))
        ims = sorted(glob.glob('img/png/*.png'))
        for b, f in zip(bckgs, ims):
            fname = os.path.basename(f)
            newname = fname.split('.')[0] + '.png'
            newname = os.path.join('img/with_bckg', newname)
            subprocess.call('composite -gravity center {} {} {}'.format(f, b, newname).split())

if False:
    m = models.Caffe(model=model, mode='gpu')
    gn_output = m.test(ims, layers=layers)
    print

    # pickle.dump(gn_output, open('resps_GoogleNet_all.npy', 'wb'))
    for sno, s in enumerate(ss[:2]):
        print sno
        dis = []
        for layer, out in gn_output.items():
            output = out.reshape((out.shape[0], -1))

            sel = np.s_[s[0]: s[1]]
            gn_dis = m.dissimilarity(output[sel])
            dis.append(gn_dis)

            hmo_dis = m.dissimilarity(hmo_output[sel])
            triu = np.triu_indices(hmo_dis.shape[0], 1)
            corr = np.corrcoef(hmo_dis[triu], gn_dis[triu])
            print layer, '%.2f' % corr[0,1]
        print
        dis = np.dstack(dis)
        np.save('dis_%s_%d.npy' % (model, sno), dis)

    for layer, output in gn_output.items():
        if layer in ['fc6', 'fc7', 'fc8']:
            np.save('resps_%s_%s.npy' % (model, layer), output)

if False:
    m = models.Caffe(model=model, mode='gpu')
    preds = m.predict(ims[ss[0][0]:ss[0][1]], topn=5)
    pickle.dump(preds, open('preds_%s.pkl' % model, 'wb'))

    import pdb; pdb.set_trace()

if False:
    preds = pickle.load(open('preds_%s.pkl' % model, 'rb'))
    preds_new = []
    for stimno, pred in enumerate(preds):
        for pno, p in enumerate(pred):
            p['stimno'] = stimno
            p['predno'] = pno
            preds_new.append(p)
    df = pandas.DataFrame(preds_new)
    with open('preds_GoogleNet.csv', 'wb') as f:
        f.writelines(df.to_csv())
    import pdb; pdb.set_trace()


    # g = sns.factorplot('model', 'accuracy', 'kind', data=lin,
    #                     kind='bar')
    # g.axes.flat[0].axhline(1/3., ls='--', c='.2')
    # base.set_vertical_labels(g)
    # base.show(pref='lin', **kwargs)

# def lin_deep(model_name='', **kwargs):
#     lin = base.lin_models(HOP2008, [model_name], **kwargs)
#     g = sns.factorplot('model', 'accuracy', 'kind', data=lin,
#                         kind='bar')
#     g.axes.flat[0].axhline(1/3., ls='--', c='.2')
#     base.show(pref='lin', **kwargs)

# def corr_models(**kwargs):
#     os.chdir(kwargs['dataset'])
#     force = False
#     try:
#         df = base.load(pref='corr', suffix='CaffeNet', **kwargs)
#     except:
#         force = True

#     if force or kwargs['force']:
#         models1 = ['pixelwise', 'shape']
#         # models2 = ['px', 'gaborjet',
#         #           'hog', 'phog', 'phow']
#         models2 = ['CaffeNet conv%d' % i for i in range(1,6)] + \
#                   ['CaffeNet fc%d' % i for i in range(6,9)]
#         df = base.corr_models(models1, models2, **kwargs)
#         df = df.rename(columns={'model1': 'model', 'model2': 'CaffeNet layer'})
#         base.save(df, pref='corr', suffix='CaffeNet', **kwargs)

#     g = sns.factorplot('CaffeNet layer', 'correlation', 'model', data=df,
#                         kind='point', palette='Set2')
#     print df
#     # chances = [1./len(np.unique(val)) for val in Stefania().dims.values()]
#     # for ax, chance in zip(g.axes.flat, chances):
#     #     ax.axhline(chance, ls='--', c='.2')
#     base.show(pref='corr', **kwargs)

def corr_all(**kwargs):
    base.compare_all(HOP2008, kind='corr', subplots=True, **kwargs)

def corr_all_layers(**kwargs):
    base.DEEP = ['CaffeNet', 'VGG-19', 'GoogleNet']
    colors = sns.color_palette('Set2')[:2]
    base.corr_all_layers(HOP2008, kinds=['px', 'shape'], colors=colors, **kwargs)

def lin_all(**kwargs):
    base.compare_all(HOP2008, kind='lin', subplots=True, **kwargs)

def group_all(**kwargs):
    base.compare_all(HOP2008, kind='group', subplots=False, **kwargs)

def group_all_layers(model_name=None, layers=None, **kwargs):
    kinds = ['px', 'shape']
    colors = sns.color_palette('Set2')[:2]
    fig, axes = sns.plt.subplots(len(base.DEEP), sharey=True, figsize=(2.5,4))
    for model, ax in zip(base.DEEP, axes):
        m = HOP2008(model_name=model, layers='all', **kwargs)
        m.dis_group(plot=False)
        # corr = base.lin_models(HOP2008, [model], kind='group', **kwargs)
        group = m.group.groupby(['kind', 'layer']).mean().reset_index()
        for kind, color in zip(kinds, colors):
            sel = group[group.kind==kind]
            ax.plot(range(len(sel)), np.array(sel.similarity), lw=3, color=color)
            ax.set_xlim([0, len(sel)-1])
        ax.set_xticklabels([])
        ax.set_title(model)
        sns.despine()

    sns.plt.ylim([0, 1])
    ax.set_yticks([0,.5,1])
    ax.set_yticklabels(['0','.5','1'])
    base.show(pref='group', suffix='all_layers', **kwargs)

def group_diff(**kwargs):
    base.compare_all(HOP2008, kind='group_diff', subplots=False, **kwargs)


    # green = sns.color_palette('Set2', 8)[0]
    # base.corr_all('px', deep, dims=dims, color=green, **kwargs)

    # orange = sns.color_palette('Set2', 8)[1]
    # base.corr_all('shape', deep, dims=dims, color=orange, **kwargs)

# def lin_all(**kwargs):
#     shallow = ['px', 'gaborjet', 'hog', 'phog', 'phow']
#     deep = ['CaffeNet', 'Places', 'HMO', 'GoogleNet']
#     models = shallow + deep
#     lin = base.lin_models(HOP2008, models, **kwargs)
#     green = sns.color_palette('Set2', 8)[0]
#     base.plot_all(lin[lin.kind=='pixelwise'], 'pixelwise', color=green,
#                   values='accuracy', pref='lin', **kwargs)

#     orange = sns.color_palette('Set2', 8)[1]
#     base.plot_all(lin[lin.kind=='shape'], 'shape', color=orange,
#                   values='accuracy', pref='lin', **kwargs)

def run(**kwargs):
    getattr(HOP2008(**kwargs), kwargs['func'])()
