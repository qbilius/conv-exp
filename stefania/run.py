import sys, os, glob, subprocess
from collections import OrderedDict

import numpy as np
import scipy.io
import pandas
import seaborn as sns
from nltk.corpus import wordnet as wn

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import stats, models

import base


class Stefania(base.Base):

    def __init__(self, *args, **kwargs):
        kwargs['skip_hmo'] = False
        super(Stefania, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        kinds = np.meshgrid(range(6), range(9))
        kinds = [kinds[1], kinds[1], kinds[0], (kinds[0] < 3).astype(int)]
        kinds = [k.ravel() for k in kinds]
        self.dims = OrderedDict()
        self.colors = OrderedDict()
        for i, (dim, kind) in enumerate(zip(['px', 'shape', 'category', 'natural-vs-manmade'], kinds)):
            self.dims[dim] = kind
            self.colors[dim] = base.COLORS[i]

        if self.filter:
            self.sel = self.filter_imagenet()
            # self.ims = [im for im,s in zip(self.ims,sel) if s]
            # self.dims = {k:v[sel] for k,v in self.dims.items()}
            # self.sel = sel
    # def mds(self):
    #     self.dissimilarity()
    #     path = os.path.join('img', 'png', '*.*')
    #     ims = sorted(glob.glob(path))
    #     super(Base, self).mds(self.dis, ims, kind='metric')
    #     self.show(pref='mds')

    def mds(self):
        p = sns.color_palette('Set2', 8)
        colors = [p[1], p[5], p[6], p[2], p[3], p[7]]
        icons = []
        ims = sorted(glob.glob('stefania/img/alpha/*.png'))
        for imno,c in enumerate(self.dims['category']):
            im = models.load_image(ims[imno], keep_alpha=True)
            mask = im[:,:,3]>0
            im[:,:,:3][mask] = colors[c][0], colors[c][1], colors[c][2]
            icons.append(im)
        super(Stefania, self).mds(icons=icons, seed=0, zoom=.15)

    def mds_outlines(self):
        p = sns.color_palette('Set2', 8)
        colors = [p[1], p[5], p[6], p[2], p[3], p[7]]
        icons = []
        ims = sorted(glob.glob('stefania/img/alpha/*.png'))
        for imno,c in enumerate(self.dims['category']):
            im = models.load_image(ims[imno], keep_alpha=True)
            # generate outlines
            mask = im[:,:,3]>0
            icon = scipy.ndimage.binary_dilation(mask, structure=np.ones((50,50)))
            icon = np.dstack([icon*colors[c][i] for i in range(3)] + [(icon>0).astype(float)])
            icon[mask] = im[mask]
            icons.append(icon)
        super(Stefania, self).mds(icons=icons, seed=0, zoom=.15)

    def corr(self):
        self.dims = OrderedDict([(k,v) for k,v in self.dims.items() if k in ['shape', 'category']])
        return super(Stefania, self).corr()

    def plot_single(self, *args, **kwargs):
        return super(Stefania, self).plot_single(ylim=[-.1,1], *args, **kwargs)

    def filter_imagenet(self):
        #myexp = Stefania(**self.kwargs)
        #myexp.set_model('caffenet')
        # import pdb; pdb.set_trace()
        df = self.pred_acc(compute_acc=False)
        sel = np.array(df.kind!='unknown')
        #sel = np.array(df.accuracy==True)
        #import pdb; pdb.set_trace()

        return sel

    # def plot_lin(self):
    #     xlabel = '%s layers' % self.model_name
    #     self.lin = self.lin.rename(columns={'layer': xlabel})
    #     colors = sns.color_palette('Set2')[1:len(self.dims)+1]

    #     for (kind, value), color in zip(self.dims.items(), colors):
    #         chance = 1. / len(np.unique(value))
    #         df = self.lin[self.lin.kind==kind]
    #         sns.set_palette([color])
    #         if self.model_name == 'GoogleNet':
    #             g = sns.factorplot(xlabel, 'accuracy', data=df,
    #                               kind='point', markers='None', ci=0)
    #             g.axes.flat[0].set_xticklabels([])

    #             # import matplotlib.lines as mlines
    #             # handles = []
    #             # for mname, color in zip(self.dims.keys(), colors):
    #             #     patch = mlines.Line2D([], [], color=color, label=mname)
    #             #     handles.append(patch)
    #             # g.axes.flat[0].legend(handles=handles, loc='best')
    #         else:
    #             g = sns.factorplot(xlabel, 'accuracy', data=df,
    #                               kind='point')
    #         g.axes.flat[0].axhline(chance, ls='--', c='.2')
    #         g.axes.flat[0].set_ylim([0,1])
    #         sns.plt.title(kind)

    #         self.show(pref='lin_%s' % kind)

    def filter_stim(self):
        with open('names.txt', 'rb') as f:
            lines = f.readlines()
        f = open('names-matched.csv', 'wb')
        for line in lines:
            name = line.strip('\r\n')#.split(',')[0]
            syns = wn.synsets('_'.join(name.split()), pos='n')
            print '~' * 40
            print
            print name
            print
            for n, syn in enumerate(syns):
                print n, '-', syn.definition()
            print
            while True:
                num = raw_input('Which definition is correct? (or type another name to check) ')
                if num == 'q':
                    f.close()
                    sys.exit()
                try:
                    num = int(num)
                except:
                    print
                    print name
                    print
                    syns += wn.synsets(num, pos='n')
                    for n, syn in enumerate(syns):
                        print n, ' - ', syn.definition()
                    print
                else:
                    if num in range(len(syns)):
                        break

            syn = syns[num]
            synid = syn.pos() + str(syn.offset()).zfill(8)
            f.write(','.join([synid, name, syn.definition()]))
        f.close()

    def gen_alpha(self):
        for fn in sorted(glob.glob('stefania/img/*.jpg')):
            fname = os.path.basename(fn)
            newname = fname.split('.')[0] + '.png'
            newfname = os.path.join('stefania/img/alpha', newname)
            fuzz = '3%'
            # if fname == 'hopStim_054.jpg':
            #     fuzz = '5%'
            # else:
            #     fuzz = '10%'
            subprocess.call(('convert {} -alpha set -channel RGBA -fuzz ' + fuzz +
                            ' -fill none -floodfill +0+0 white -blur 1x1 {}').format(fn, newfname).split())

def gen_sil(**kwargs):
    for fn in sorted(glob.glob('img/*.jpg')):
        fname = os.path.basename(fn)
        newname = fname.split('.')[0] + '.png'
        newfname = os.path.join('img/sil_new', newname)
        fuzz = '3%'
        subprocess.call(('convert {} -alpha set -channel RGBA -fuzz ' + fuzz +
                        ' -fill none -floodfill +0+0 white -blur 1x1 '
                        '-alpha extract -negate {}').format(fn, newfname).split())

def corr_models(mods1_dis, mods2_dis):
    df = []
    for mods1_label, mods1_data in mods1_dis.items():
        inds = np.triu_indices(mods1_data.shape[0], k=1)
        for mods2_label, mods2_data in mods2_dis.items():
            corr = np.corrcoef(mods1_data[inds], mods2_data[inds])[0,1]
            df.append([mods1_label, mods2_label, corr])
    df = pandas.DataFrame(df, columns=['perception', 'models', 'correlation'])
    df = stats.factorize(df)
    sns.factorplot('perception', 'correlation', 'models',
                   data=df, kind='bar')
    return df

def corrplot(mod_dis):
    df_model = []
    for label, data in mod_dis.items():
        inds = np.triu_indices(data.shape[0], k=1)
        df_model.append(data[inds])

    df_model = pandas.DataFrame(np.array(df_model).T,
                                columns=mod_dis.keys())

    sns.corrplot(df_model)

def corr_neural_model_avg(neural_dis, mod_dis):
    df = []
    avg = [['pixelwise',
            ('BA17', 'BA18', 'TOS', 'postPPA',
            'LOTCobject', 'LOTCface')],

            ['shape',
            ('LOTCbody', 'LOTChand',
            'VOTCobject', 'VOTCbody/face')],

            ['animate/inanimate',
            ('LOTCobject', 'LOTCface', 'LOTCbody', 'LOTChand',
            'VOTCobject', 'VOTCbody/face')],

            ['nat/artifact', ('IPS', 'SPL',
             'IPL', 'DPFC')]
           ]

    for a, rois in avg:
        for mod_label, mod_data in mod_dis.items():
            for k in xrange(neural_dis.values()[0].shape[-1]):
                nd_avg = []
                for neural_label, neural_data in neural_dis.items():
                    if neural_label in rois:
                        nd_avg.append(neural_data[:,:,k])
                nd_avg = np.average(nd_avg, axis=0)

                inds = np.triu_indices(nd_avg.shape[0], k=1)
                corr = np.corrcoef(nd_avg[inds],
                                   mod_data[inds])[0,1]

                df.append([a, mod_label, k, corr])
    df = pandas.DataFrame(df, columns=['neural', 'model',
                                       'subjid', 'correlation'])
    df = stats.factorize(df)
    sns.factorplot('neural', 'correlation', 'model', data=df, kind='bar')
    return df

def corr_neural_model(neural_dis, mod_dis):
    df = []

    for mod_label, mod_data in mod_dis.items():
        for k in xrange(neural_dis.values()[0].shape[-1]):
            nd_avg = []
            for neural_label, neural_data in neural_dis.items():
                inds = np.triu_indices(mod_data.shape[0], k=1)
                corr = np.corrcoef(neural_data[:,:,k][inds],
                                   mod_data[inds])[0,1]

                df.append([neural_label, mod_label, k, corr])
    df = pandas.DataFrame(df, columns=['neural', 'model',
                                       'subjid', 'correlation'])
    df = stats.factorize(df)
    sns.factorplot('neural', 'correlation', 'model', data=df, kind='bar')
    return df


class Compare(base.Compare):
    def __init__(self, *args):
        super(Compare, self).__init__(*args)

    # def corr(self):
    #
    #     return super(Compare, self).corr()

def report(**kwargs):
    html = kwargs['html']
    html.writeh('Stefania', h='h1')

    html.writeh('MDS', h='h2')
    kwargs['layers'] = 'output'
    kwargs['task'] = 'run'
    kwargs['func'] = 'mds'
    myexp = Stefania(**kwargs)
    myexp.set_model('googlenet')
    myexp.mds()

    html.writeh('Correlation', h='h2')

    html.writeh('Original stimuli', h='h3')
    kwargs['layers'] = 'all'
    kwargs['task'] = 'run'
    kwargs['func'] = 'corr'
    myexp = Stefania(**kwargs)

    for depth, model_name in myexp.models:
        myexp.set_model(model_name)
        if depth != 'shallow':
            myexp.corr()

    kwargs['layers'] = 'output'
    kwargs['task'] = 'compare'
    kwargs['force'] = False
    kwargs['forceresps'] = False
    myexp = Stefania(**kwargs)
    Compare(myexp).corr()

    html.writeh('Only in the ImageNet', h='h3')
    kwargs['layers'] = 'output'
    kwargs['filter'] = True
    myexp = Stefania(**kwargs)
    myexp.skip_hmo = True
    myexp.set_models()
    Compare(myexp).corr()

    html.writeh('Silhouettes', h='h3')
    kwargs['layers'] = 'output'
    kwargs['subset'] = 'sil'
    kwargs['filter'] = False
    myexp = Stefania(**kwargs)
    myexp.skip_hmo = True
    myexp.set_models()
    Compare(myexp).corr()

    html.writeh('Silhouettes in ImageNet', h='h3')
    kwargs['layers'] = 'output'
    kwargs['subset'] = 'sil'
    kwargs['filter'] = True
    myexp = Stefania(**kwargs)
    myexp.skip_hmo = True
    myexp.set_models()
    Compare(myexp).corr()
