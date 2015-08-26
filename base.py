import os, sys, glob, subprocess, copy, itertools
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import scipy.misc, scipy.io, scipy.linalg
import pandas
# from PIL import Image, ImageDraw

import sklearn
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import StratifiedKFold

import skimage.draw

# import scikits.bootstrap as boot
from nltk.corpus import wordnet as wn

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import seaborn as sns

sys.path.insert(0, '../psychopy_ext')
from psychopy_ext import models, stats

SHALLOW = ['px', 'gaborjet', 'hog', 'phog', 'phow']
DEEP = ['CaffeNet', 'HMO', 'VGG-19', 'GoogleNet']


def _check_force(pref, plot=True):
    def decorator(func):
        def func_wrapper(self):
            setup_result = self._setup_model()
            if self.force and setup_result:
                func(self)
            else:
                data = self.load(pref)
                if data is not None:
                    setattr(self, pref, data)
                else:
                    func(self)
            if plot:
                self.plot(pref)
        return func_wrapper
    return decorator

def style_plot(plot_func):
    def func_wrapper(self, df, values, aspect=1, **kwargs):

        if self.model_name == 'GoogleNet':
            aspect = 3
        elif self.model_name == 'VGG-19':
            aspect = 2
        else:
            aspect = 1

        g = plot_func(self, df, values, aspect=aspect,**kwargs)

        if 'ci_low' in df.columns:
            plot_ci(df)

        if 'accuracy' in df.columns:
            for value in self.dims.values():
                plot_chance(value)

        if self.model_name in ['GoogleNet', 'VGG-19']:
            labels = g.axes.flat[0].get_xticklabels()
            for label in labels:
                if len(label.get_text()) > 3:
                    label.set_ha('right')
                    label.set_rotation(30)
            if self.model_name == 'GoogleNet':
                sns.plt.subplots_adjust(bottom=.25)

    return func_wrapper

def plot_ci(df, what=['Line2D']):

    # lines = sns.plt.gca().get_lines()
    children = sns.plt.gca().get_children()
    colors = []
    for child in children:
        spl = str(child).split('(')[0]
        if spl in what:
            if spl == 'Line2D':
                if child.get_color() not in colors:
                    colors.append(child.get_color())
            else:
                colors.append('.15')

    if 'kind' in df:
        for kind, color in zip(df.kind.unique(), colors):
            sel = df[df.kind==kind]
            for r, (rowno, row) in enumerate(sel.iterrows()):
                sns.plt.plot([r,r], [row.ci_low, row.ci_high], color=color,
                             lw=sns.mpl.rcParams['lines.linewidth']*1.8)
    else:
        for r, (rowno, row) in enumerate(df.iterrows()):
            sns.plt.plot([r,r], [row.ci_low, row.ci_high], color=colors[0],
                         lw=sns.mpl.rcParams['lines.linewidth']*1.8)

def plot_chance(value):
    # try:
    #     g.axes.flat
    # except:
    #     axes = g.axes
    # else:
    #     axes = g.axes.flat
    # for ax, value in zip(g.axes.flat, dims.values()):
    chance = 1. / len(np.unique(value))
    sns.plt.axhline(chance, ls='--', c='.15')

def recolor(g, palette=None, what=[]):
    # ['Line2D', 'Rectangle', 'Path']
    for ax, colors in zip(g.axes.flat, palette):
        children = ax.get_children()
        if isinstance(colors, list):
            colors = itertools.cycle(colors)
        for child in children:
            spl = str(child).split('(')[0]
            if spl in what:
                try:
                    color = colors.next()
                except:
                    color = colors
                child.set_color(color)
        #     label = child.get_label()
        #     try:
        #         if label.startswith('_collection'):
        #             points = child
        #             break
        #     except:
        #         pass
        # points.set_color(color)

        # lines = ax.get_lines()
        # for line in lines:
        #     line.set_color(color)

class Base(models.Model):

    def __init__(self, model_name='CaffeNet', layers='all',
                 dataset='', subset=None, mode='gpu',
                 savedata=True, savefig='', saveresps=False,
                 force=False, filter=False, task=None, func=None):
        super(Base, self).__init__()

        # if model_name in models.CAFFE_MODELS:
        #     self.m = models.Model()
        #     self.m.dissimilarity = models.Caffe.dissimilarity
        # else:
        #     try:
        #         self.m = models.KNOWN_MODELS[model_name]()
        #     except:
        #         print 'WARNING: cannot find the model'

        self.model_name = model_name
        self.layers = layers
        self.dataset = dataset
        if subset is not None:
            path = os.path.join('img', subset, '*.*')
        else:
            path = os.path.join('img', '*.*')
        self.subset = subset
        self.ims = sorted(glob.glob(path))
        self.savedata = savedata
        self.savefig = savefig
        self.saveresps = saveresps
        self.force = force
        self.filter = filter
        self.task = task
        self.func = func
        if self.model_name in ['GoogleNet', 'VGG-19']:
            self.mode = 'cpu'  # not enough memory on my GPU!
            print 'forced mode to CPU'
        else:
            self.mode = mode

    def _setup_model(self):
        if self.model_name in models.CAFFE_MODELS:
            self.m = models.Caffe(model=self.model_name,
                                  layers=self.layers, mode=self.mode)
        elif self.model_name == 'Places':
            self.m = models.Caffe(model='Places205-CNN', mode=self.mode,
                                  layers=self.layers)
        elif self.model_name == 'VGG-19':
            self.m = models.Caffe(model='VGG_ILSVRC_19_layers',
                                  layers=self.layers, mode=self.mode)
        elif self.model_name in models.KNOWN_MODELS:
            self.m = models.KNOWN_MODELS[self.model_name](layers=self.layers)
        else:
            return False
        return True

    def filter_imagenet(self):
        return None

    @_check_force('resps')
    def classify(self):
        setup_result = self._setup_model()
        if setup_result == False:
            print ('%s is not available for generating responses' %
                    self.model_name)
            self.resps = self.load('resps')
            if self.resps is None:
                raise Exception('no response file found for %s' %
                                self.model_name)
            return
        output = self.m.run(self.ims, return_dict=True)

        self.resps = OrderedDict()
        for layer, out in output.items():
            self.resps[layer] = out.reshape((out.shape[0], -1))
        #self._save('resps')

    @_check_force('dis')
    def dissimilarity(self):
        self.classify()
        self.dis = super(Base, self).dissimilarity(self.resps,
                                                   kind='corr')
        self.save('dis')

    def plot_dis(self):
        sns.heatmap(self.dis[self.dis.keys()[0]])
        self.show('dis')

    @_check_force('preds')
    def predict(self):
        setup_result = self._setup_model()
        if setup_result == False:
            raise Exception('%s is not available for generating predictions' %
                    self.model_name)
        self.preds = self.m.predict(self.ims, topn=5)
        self.save('preds')

    def clustermap(self):
        self.dissimilarity()
        sns.clustermap(self.dis.values()[0])
        self.show()

    def mds(self):
        self.dissimilarity()
        super(Base, self).mds(self.dis, self.ims, kind='metric')
        self.show(pref='mds')

    def pred_acc(self, compute_acc=True):
        if compute_acc:
            self.predict()
        imagenet_labels = self.synsets_from_txt('../synset_words.txt')
        dataset_labels = self.synsets_from_csv(self.dataset + '.csv')
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
                for p in self.preds[no]:
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

    # def pred_acc(self, synsnod):
    #     synimg = self.synsets_from_txt('../synset_words.txt')
    #     # synsnod = self.synsets_from_csv(self.dataset + '.csv')
    #     df = pandas.DataFrame.from_dict(synsnod)
    #     df['imgid'] = ''
    #     df['imdnames'] = ''
    #     df['kind'] = 'unknown'
    #     df['accuracy'] = np.nan
    #
    #     for predno, (pred, snod) in enumerate(zip(self.preds, synsnod)):
    #         # get all more specific synsets of the correct resp
    #         snodhypos = set([i for i in snod['synset'].closure(lambda s:s.hyponyms())])
    #         snodhypos = snodhypos.union([snod['synset']])
    #         for p in pred:
    #             psyn = wn._synset_from_pos_and_offset(p['synset'][0], int(p['synset'][1:]))
    #             # check if the prediction is exact
    #             # or at least more specific than the correct resp
    #             if psyn in snodhypos:
    #                 df.loc[predno, 'accuracy'] = True
    #                 df.loc[predno, 'imgid'] = p['synset']
    #                 df.loc[predno, 'imgnames'] = p['label']
    #
    #                 if p['synset'] == df.loc[predno, 'id']:
    #                     df.loc[predno, 'kind'] = 'exact'
    #                 else:
    #                     df.loc[predno, 'kind'] = 'superordinate'
    #                 break
    #         else:  # the response is totally incorrect
    #             for img in synimg:
    #                 # but the model knows this synset
    #                 if img['synset'] in snodhypos:
    #                     df.loc[predno, 'accuracy'] = False
    #                     df.loc[predno, 'imgnames'] = '; '.join([p['label'] for p in pred])
    #
    #                     if img['id'] == df.loc[predno, 'id']:
    #                         df.loc[predno, 'kind'] = 'exact'
    #                     else:
    #                         df.loc[predno, 'kind'] = 'superordinate'
    #                     break
    #     return df

    def synsets_from_csv(self, fname):
        with open(fname, 'rb') as f:
            lines = f.readlines()
        df = []
        for line in lines:
            spl = line.strip('\n').split(',')
            try:
                synset = wn._synset_from_pos_and_offset(spl[0][0], int(spl[0][1:]))
            except:
                import pdb; pdb.set_trace()

            df.append({'id':spl[0], 'names':spl[1], 'synset':synset})
        # df = pandas.DataFrame(df, columns=['id', 'names', 'synset'])
        return df

    def synsets_from_txt(self, fname):

        with open(fname, 'rb') as f:
            lines = f.readlines()
        df = []
        for line in lines:
            w = line.split()[0]
            descr = line.strip('\r\n').replace(w+' ', '')
            synset = wn._synset_from_pos_and_offset(w[0], int(w[1:]))
            df.append({'id':w, 'names':descr, 'synset':synset})
        # df = pandas.DataFrame(df, columns=['id', 'names', 'synset'])
        return df

    def selfcorr(self):
        self.dissimilarity()
        df = []
        for layer, dis in self.dis.items():
            inds = np.triu_indices(dis.shape[0], k=1)
            df.append(dis[inds])
        df = pandas.DataFrame(np.array(df).T, columns=self.dis.keys())
        sns.heatmap(df.corr())
        sns.plt.title('correlations between layers')
        self.show(pref='selfcorr')

    def partial_corr(self):
        self.dissimilarity()
        df = []
        for layer, dis in self.dis.items():
            inds = np.triu_indices(dis.shape[0], k=1)
            df.append(dis[inds])
        # df = pandas.DataFrame(df, columns=self.dis.keys())
        df = np.array(df)
        self._partial_corr(df, keys=self.dis.keys())
        self.plot_partial_corr()

    def _partial_corr(self, d, keys=None):
        nc = len(d)
        # c = np.zeros((nc, nc))
        c = []
        # c[np.diag_indices(nc)
        if keys is None:
            keys = range(1,nc+1)
        for i in xrange(nc):
            c.append([keys[i], keys[i], 1])
            # c[r1,r1] = 1
            for j in xrange(i+1, nc):
                idx = np.ones(nc, dtype=np.bool)
                idx[i] = False
                idx[j] = False

                beta_i = scipy.linalg.lstsq(d[idx].T, d[i].T)[0]
                beta_j = scipy.linalg.lstsq(d[idx].T, d[j].T)[0]

                res_i = d[i].T - d[idx].T.dot(beta_i)
                res_j = d[j].T - d[idx].T.dot(beta_j)

                # c[r1,r2] = c[r2,r1] = np.corrcoef(res_i, res_j)[0,1]
                cc = np.corrcoef(res_i, res_j)[0,1]
                c.append([keys[i], keys[j], cc])
                c.append([keys[j], keys[i], cc])
        self.pcorr = pandas.DataFrame(c,
                    columns=['layers1', 'layers2', 'correlation'])
        # return c

    def plot_partial_corr(self):
        pcorr = self.pcorr.pivot(index='layers1', columns='layers2',
                                 values='correlation')
        sns.heatmap(pcorr)
        sns.plt.title('partial correlations between layers')
        self.show('pcorr')

    def plot_rec_fields(self):
        self.ims = [self.ims[0]]
        self.classify()

        sns.set_style('white')
        for layer in self.resps.keys():
            feat = self.m.net.blobs[layer].data[0, :36]
            self.vis_square(feat, layer=layer, padval=1)

    def vis_square(self, data, layer='', padsize=1, padval=0):
        """
        From Caffe
        """
        data -= data.min()
        data /= data.max()

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]),
                   (0, padsize), (0, padsize))
        padding += ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                      constant_values=(padval, padval))

        # tile the filters into an image
        data = data.reshape((n,n) + data.shape[1:])
        data = data.transpose((0,2,1,3) + tuple(range(4, data.ndim+1)))
        data = data.reshape((n * data.shape[1],
                             n * data.shape[3]) + data.shape[4:])

        plt.imshow(data)
        plt.axis('off')
        self.show(pref=layer+'2')

    def save(self, pref):
        data = getattr(self, pref)
        save(data, pref=pref, suffix=self.model_name, **self.kwargs)

    def load(self, pref):
        try:
            data = eval('self.' + pref)
        except:
            data = load(pref=pref, suffix=self.model_name, **self.kwargs)

        return data

    def show(self, pref):
        show(pref=pref, suffix=self.model_name, **self.kwargs)

    def plot(self, pref, **kwargs):
        if self.func.startswith(pref) and not self.func.startswith(pref+'_') \
        and self.task != 'compare':
            getattr(self, 'plot_' + pref)(**kwargs)
            self.show(pref)

    def plot_single_model(self, df, subplots=True, aspect=1,
                          colors=None, **kwargs):
        if 'correlation' in df:
            values = 'correlation'
        elif 'accuracy' in df:
            values = 'accuracy'
        elif 'similarity' in df:
            values = 'similarity'
        elif 'preference for shape' in df:
            values = 'preference for shape'
        else:
            raise Exception('values=%s not recognized')

        if 'kind' in df:
            hue = 'kind'
        else:
            hue = None
            subplots = False

        if subplots:
            if colors is None:
                colors = sns.color_palette('Set2')[:len(self.dims)]
            for kind, color in zip(df.kind.unique(), colors):
                self._plot_single_model(df[df.kind==kind], values,
                            color=color, aspect=aspect, hue=None,
                            **kwargs)
                sns.plt.title(kind)
        else:
            self._plot_single_model(df, values, aspect=aspect,
                                    hue=hue, color=colors, **kwargs)

    @style_plot
    def _plot_single_model(self, df, values, color=None, hue=None, aspect=1, **kwargs):
        g = sns.factorplot('%s layer' % self.model_name,
                values, data=df, hue=hue, ci=0,
                kind='point', color=color, aspect=aspect)
        sns.plt.ylim([-.1, 1.1])
        return g

    @_check_force('lin')
    def linear_clf(self):
        self.classify()
        df = []
        kinds = []
        for kind, dim in self.dims.items():
            y = dim.ravel()
            lin = super(Base, self).linear_clf(self.resps, y)
            df.append(lin)
            kinds.extend([kind]*len(lin))
        lin = pandas.concat(df, ignore_index=True)#keys=dims.keys())
        # lin.index.names = ['kind', None]
        # lin = lin.reset_index(level='kind')
        lin['kind'] = kinds
        self.lin = stats.factorize(lin)
        self.save('lin')

    def plot_lin(self, subplots=True):
        xlabel = '%s layer' % self.model_name
        self.lin = self.lin.rename(columns={'layer': xlabel})
        self.plot_single_model(self.lin, subplots=subplots)

    def corr(self, models, subplots=True, **kwargs):
        self.corr = corr_models(self.__class__, models,
                                [self.model_name], **kwargs)
        self.plot('corr', subplots=subplots, **kwargs)

    def plot_corr(self, subplots=False, **kwargs):
        self.corr = self.corr.rename(columns={'model1': 'kind',
                             'model2': '%s layer' %self.model_name})
        self.plot_single_model(self.corr, subplots=subplots, **kwargs)
        plot_ci(self.corr)

class Shape(Base):

    def _perceptron(self, dims):
        print
        print '### Perceptron ###'

        df = []
        for kind, dim in dims.items():

            n_folds = len(y) / len(np.unique(y))
            for layer, resps in self.resps.items():
                # n_itern determined empirically to give a perfect training
                # accuracy
                perc = Perceptron(eta0=.25, n_iter=25)
                svm = sklearn.svm.LinearSVC()
                cv = sklearn.cross_validation.StratifiedKFold(y,
                        n_folds=n_folds)
                # scores = sklearn.cross_validation.cross_val_score(perc,
                #             resps, y, cv=cv)

                # from scikit-learn docs:
                # need not match cross_val_scores precisely!!!
                # preds = sklearn.cross_validation.cross_val_predict(svm,
                #      resps, y, cv=cv)

                preds = []
                for traini, testi in cv:
                    for nset in range(1, n_folds+1):
                        trainj = []
                        for j in np.unique(y):
                            sel = np.array(traini)[y[traini]==j]
                            sel = np.random.choice(sel, nset).tolist()
                            trainj.extend(sel)
                    svm.fit(resps[trainj], y[trainj])
                    preds.append(svm.predict(resps[testi], y[testi]))
                # confusion[kind][layer] = sklearn.metrics.confusion_matrix(y, preds)
                # print confusion[kind][layer]
                # perc.fit(resps, y)
                # print layer, np.mean(scores) #perc.score(resps, y)
                # import pdb; pdb.set_trace()
                # for score in scores:
                #     df.append([kind, layer, score])
                for yi, pred in zip(y, preds):
                    df.append([kind, layer, yi, pred, yi==pred])
                # preds = perc.predict(resps)
                # for pred, cat, shape in zip(preds, cats.ravel(), shapes.ravel()):
                #     df.append([layer, cat, shape, pred==cat])

        # self.df = pandas.DataFrame(df, columns=['layer', 'category',
        #                                   'shape', 'accuracy'])


    @_check_force('ssize')
    def ssize(self, dims):
        self.classify()
        print
        print '### Set size ###'

        df = []
        for kind, dim in dims.items():
            y = dim.ravel()
            n_folds = len(y) / len(np.unique(y))
            print kind
            for layer, resps in self.resps.items():
                # accuracy
                perc = Perceptron(eta0=.25, n_iter=25)
                svm = sklearn.svm.LinearSVC()

                preds = []
                for nset in range(1, n_folds):
                    print nset
                    cv = sklearn.cross_validation.StratifiedKFold(y,
                        n_folds=n_folds)
                    for traini, testi in cv:
                        trainj = []
                        for j in np.unique(y):
                            sel = np.array(traini)[y[traini]==j]
                            sel = np.random.choice(sel, nset, replace=False).tolist()
                            trainj.extend(sel)
                        svm.fit(resps[trainj], y[trainj])
                        preds = svm.predict(resps[testi])
                        for yi, pred in zip(y[testi], preds):
                            df.append([kind, layer, nset, yi, pred,
                                      yi==pred])
        self.perc = pandas.DataFrame(df, columns=['kind', 'layer',
                                'set_size',
                                'actual', 'predicted', 'accuracy'])
        self.ssize = stats.factorize(self.perc)
        self._save(self.ssize, 'ssize')

    def plot_ssize(self):
        colors = sns.color_palette('Set2')[1:len(self.dims)+1]
        # agg = stats.aggregate(self.ssize, yerr='set_size', cols='kind',
        #                       values='accuracy')
        if self.layers not in [None, 'all']:
            self.ssize = self.ssize[self.ssize.layer==self.layers]
            self.model_name += '_' + self.layers
        agg = self.ssize.pivot_table(index='set_size', columns='kind',
                                      values='accuracy')
        self.set_size = self.ssize.rename(columns={'set_size':'number of training examplars'})

        import scipy.optimize
        def func(x, a, b):
            return a * np.log(x) + b

        for (kind, value), color in zip(self.dims.items(), colors):
            chance = 1. / len(np.unique(value))
            # df = self.set_size[self.set_size.kind==kind]
            df = agg[kind].reset_index()
            df = df.rename(columns={'set_size':'number of training examplars', kind:'accuracy'})
            sns.set_palette([color])

            g = sns.lmplot('number of training examplars', 'accuracy',
                               data=df, x_estimator=np.mean,
                               logx=True, x_ci=0)

            ax = g.axes.flat[0]
            ax.axhline(chance, ls='--', c='.2')
            ax.set_xlim([.5, len(agg.index) + .5])
            ax.set_ylim([0,1])

            y = agg[kind][~np.isnan(agg[kind])].values
            x = agg[kind].index[~np.isnan(agg[kind])].values
            popt, pcov = scipy.optimize.curve_fit(func, x, y)
            perf = np.exp((1 - popt[1]) / popt[0])
            print kind, '%d' % round(perf)
            ax.set_title(kind + ', perfect performance with %d images' % round(perf))

            self.show(pref='ssize_%s' % kind)

    def plot_confusion(self, layer):
        def confusion(data, color):
            conf = sklearn.metrics.confusion_matrix(data.actual,
                                                    data.predicted)
            return sns.heatmap(conf, cmap=cmap)
        df = self.df[self.perc.layer==layer]

        g = sns.FacetGrid(df, col='kind', size=3,
                          sharex=False, sharey=False)
        cmap = sns.cubehelix_palette(reverse=True, as_cmap=True)
        g.map_dataframe(confusion)
        self.show()

    def nat_man_conv(self):
        with open('multilabel.html') as f:
            html_doc = f.read()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_doc)
        df = {}
        with open('multicats.txt') as f:
            lines = f.readlines()
        f = open('multilabel.txt', 'wb')
        for cn, cat in enumerate(soup.find_all('div',id='image-table')):
            first = False
            for img in cat.find_all('div', class_='row'):
                descr = img.find_all('font')
                f.write(descr[0].text + ': ')
                f.write(','.join([d.text for d in descr[1:]]))
                f.write('\n')
                if first:
                    if lines[cn].strip('\r\n') != descr[1].text:
                        print lines[cn].strip('\r\n'), descr[1].text
                    first = False
        f.close()

    def nat_man(self):
        if self._check_force('df_nat_man'):
            self._nat_man()
        self.plot_nat_man()

    def _nat_man(self):
        # from nltk.corpus import wordnet as wn
        with open('multilabel.txt') as f:
            lines = f.readlines()

        with open('multicats.txt') as f:
            allcats = f.readlines()
            allcats = [d.strip('\r\n').split(',') for d in allcats]
            allcats = OrderedDict(allcats)

        # artifact = [wn.synsets('artifact')[0]]
        # living = [wn.synsets('living_thing')[0],
        #           wn.synset('natural_object.n.01')]
        # hyper = lambda s:s.hypernyms()

        inds = []
        for cat1 in cats.keys():
            for cat2 in cats.keys():
                inds.append((cat1, cat2, cats[cat1], cats[cat2]))
        # df = pandas.DataFrame(inds, columns=['cat1', 'cat2',
        #                                      'kind1', 'kind2', 'count'])
        multi = pandas.MultiIndex.from_tuples(inds,
                                              names=['cat1', 'cat2',
                                                    'kind1', 'kind2'])

        df = pandas.Series(np.zeros(len(multi), dtype=int), index=multi)
        # multi = pandas.MultiIndex.from_product([cats.keys(),
        #                                          cats.keys()])
        # df = np.zeros((len(cats),len(cats)), dtype=int)
        # df = pandas.DataFrame(df, index=cats.keys(), columns=cats.keys())
        used_ims = []
        # g = open('multilabels_unq.txt', 'wb')
        df = []
        for n, line in enumerate(lines):
            print n,
            linestr = line.strip('\n\r')
            im, cats = linestr.split(': ')
            if not im in used_ims:
                # g.write(line)
                cats = cats.split(',')
                used_ims.append(im)
                for cat1 in cats:
                    for cat2 in cats:
                        if cat1 != cat2:
                            df[cat1, cat2] += 1
                        else:
                            df[cat1, cat2] = np.nan
                            # df.loc[np.logical_and(df.cat1==cat1,
                            #     df.cat2==cat2), 'count'] = np.nan
                        # else:
                        #     df.loc[np.logical_and(df.cat1==cat1,
                        #         df.cat2==cat2), 'count'] += 1

        df = pandas.DataFrame(df, columns=['im','natural','manmade'])
        df = df.reset_index()
        df = df.rename(columns={0:'count'})
        df = df.sort(['kind1', 'kind2'])
        self.df_nat_man = df
        self._save(df, 'df_nat_man')

    def plot_nat_man(self):
        df = self.df_nat_man
        df = df[np.logical_and(df.kind1!='food', df.kind2!='food')]
        self._plot_nat_man(df, 'nat_man')
        dfp = df[np.logical_and(df.cat1!='person', df.cat2!='person')]
        self._plot_nat_man(dfp, 'nat_man_nop')
        # for cat in np.unique(df):
        #     cat_syns = wn.synsets(cat, pos='n')
        #     tmp1 = []
        #     tmp2 = []
        #     for syn in cat_syns:
        #         hyps = set([i for i in syn.closure(hyper)])
        #         tmp1.append(any(art in hyps for art in artifact))
        #         tmp2.append(any(liv in hyps for liv in living))
        #     if any(tmp1):
        #         if any(tmp2):
        #             import pdb; pdb.set_trace()
        #         man.append(cat)
        #     elif any(tmp2):
        #         if any(tmp1):
        #             import pdb; pdb.set_trace()
        #         nat.append(cat)
        #     else:
        #         print cat
        #         # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

    def _plot_nat_man(self, df, pref):
        agg = df.pivot_table(index=['kind1','cat1'],
                             columns=['kind2','cat2'],
                             values='count')
        freq = agg.sum(level='kind1') / agg.sum()
        freq = freq.T.stack(level='kind1').reset_index()
        freq = freq.rename(
            columns={'kind1':'proportion of ... objects in it',
            'kind2':'given that an image has a ... object',
            0:'probability'})
        sns.factorplot('given that an image has a ... object',
                       'probability',
                       'proportion of ... objects in it',
                       data=freq, kind='bar')
        mn = freq.mean(axis='index', level=0)
        print mn
        self.show(pref)


class GenSecondOrder(object):
    def __init__(self, save=True,
                 savepath='img/{kind}_{num}.png'):

        self.save = save
        self.savepath = savepath
        self.num = 0

    def gen(self, mask=(50, 100), pos=None, shape=(256,256), kind='orientation', savename=None):
        try:
            mask = np.array(mask)
        except:
            raise ValueError('mask must be a tuple or array-like')

        if mask.size == 2:
            fig_sh = tuple(mask)
            self.mask = np.zeros(shape)
            fig = np.ones(fig_sh)
            if pos is None:
                sel = np.s_[(shape[0]-fig_sh[0])/2: (shape[0]+fig_sh[0])/2,
                            (shape[1]-fig_sh[1])/2: (shape[1]+fig_sh[1])/2]
            else:
                sel = np.s_[(shape[0]-fig_sh[0])/2 + pos[0]: (shape[0]+fig_sh[0])/2 + pos[0],
                            (shape[1]-fig_sh[1])/2 + pos[1]: (shape[1]+fig_sh[1])/2 + pos[1]]
            self.mask[sel] = 1
        else:
            self.mask = mask

        self.mask = self.mask.astype(bool)
        self.shape = self.mask.shape

        if kind == 'orientation':
            angle = np.random.randint(0,180)
            imb = self.lamme(ori=angle)
            imf = self.lamme(ori=angle-90)
        elif kind == 'luminance':
            imf = np.ones(self.shape)
            imb = np.zeros(self.shape)
        elif kind == 'offset':
            angle = np.random.randint(0,180)
            imb, imf = self.offset(ori=angle)
        elif kind == 'offset_dense':
            angle = 45
            imb = self.offset_dense(ori=angle)
            imf = self.offset_dense(ori=angle)
        else:
            raise ValueError('{} not recognized'.format(kind))

        im = imb.copy()
        im[self.mask] = imf[self.mask]

        if self.save:
            if savename is not None:
                scipy.misc.imsave(savename, im)
            else:
                scipy.misc.imsave(self.savepath.format(kind=kind,
                                  num=self.num), im)
            self.num += 1
        else:
            return im

    def offset(self, ori=45, spacing=2, sf_px=5, phase=180):
        ori = np.deg2rad(ori)
        im = np.zeros(self.shape).astype(int)
        xx,yy = np.meshgrid(np.arange(im.shape[0]),
                            np.arange(im.shape[1]), indexing='xy')

        rotj1 = yy*np.sin(ori) + xx*np.cos(ori)
        grating1 = np.cos(2*np.pi*rotj1/sf_px + spacing*np.deg2rad(phase))
        grating1wide = np.cos(2*np.pi*rotj1/(spacing*sf_px) + np.deg2rad(phase))
        grating1 = np.logical_and(grating1>0, grating1wide>0)

        rotj2 = yy*np.sin(ori) + xx*np.cos(ori)
        grating2 = np.cos(2*np.pi*rotj2/sf_px)
        grating2wide = np.cos(2*np.pi*rotj2/(spacing*sf_px))
        grating2 = np.logical_and(grating2>0, grating2wide>0)

        return grating1, grating2

    def offset_dense(self, ori=45, linewidth=1):
        im = np.zeros(self.shape).astype(int)

        for it in xrange(im.size/300):
            length = 300 #np.random.randint(15,25)
            pos1 = np.random.randint(self.shape[0])
            pos2 = np.random.randint(self.shape[1])

            add1 = -np.sin(np.radians(ori)) * length/2
            add2 = np.cos(np.radians(ori)) * length/2

            for w in xrange(linewidth):
                rr, cc = skimage.draw.line(int(pos1-add1+w),
                                           int(pos2-add2+w),
                                           int(pos1+add1-w),
                                           int(pos2+add2-w))
                for r, c in zip(rr, cc):
                    try:
                        im[r,c] = 255
                    except:
                        pass

        return im

    def lum(self):
        self.im[self.mask] = 1

        if save:
            scipy.misc.imsave(self.savepath.format(name='luminance', num=self.num), self.im)
            self.num += 1
        else:
            scipy.misc.imshow(im)

    def lamme(self, ori=45, linewidth=1):
        im = np.zeros(self.shape).astype(int)
        # imp = Image.fromarray(im)
        # draw = ImageDraw.Draw(im)

        # if left:
        #     s = 1
        # else:
        #     s = -1

        for it in xrange(im.size/70):
            length = np.random.randint(15,25)
            pos1 = np.random.randint(self.shape[0])
            pos2 = np.random.randint(self.shape[1])

            add1 = -np.sin(np.radians(ori)) * length/2
            add2 = np.cos(np.radians(ori)) * length/2

            for w in xrange(linewidth):
                rr, cc = skimage.draw.line(int(pos1-add1+w),
                                           int(pos2-add2+w),
                                           int(pos1+add1-w),
                                           int(pos2+add2-w))
                for r, c in zip(rr, cc):
                    try:
                        im[r,c] = 255
                    except:
                        pass
            # for i in range(-length/2,length/2):
            #     try:
            #         im[pos1+i, pos2+s*i] = 1
            #         im[pos1+i+1, pos2+s*i] = 1
            #     except:
            #         pass

        return im


def load(pref='', dataset='', subset=None, suffix='',
         layers=[], model_name='', **kwargs):

    name = '_'.join(filter(None, [pref, dataset, subset, suffix]))
    try:
        data = pickle.load(open(name+'.pkl', 'rb'))
    except:
        try:
            data = scipy.io.loadmat(open(name+'.mat', 'rb'))
        except:
            print 'tried loading from', name
            return None
        else:
            print 'loaded from', name+'.mat'
            data = OrderedDict([(model_name, data[model_name])])
    else:
        print 'loaded from', name + '.pkl'

    data = filter_layers(data, layers, pref)
    return data

def filter_layers(data, layers, pref):
    if pref == 'dis':
        if layers is None:
            layer = data.keys()[-1]
            data = OrderedDict([(layer, data[layer])])
            print 'using layer %s only' % layer
        elif isinstance(layers, str):
            if layers != 'all':
                try:
                    data = OrderedDict([(layers, data[layers])])
                except:
                    print 'layers %s not found, reclassifying' % layers
                    return None
        else:
            try:
                data = OrderedDict([(l,data[l]) for l in layers])
            except:
                print 'layers %s not found, reclassifying' % layers
                return None

    elif pref in ['lin', 'nap']:
        if layers is None:
            layer = data.layer.unique()[-1]
            data = data[data.layer==layer]
            print 'using layer %s only' % layer
        elif isinstance(layers, str):
            if layers == 'all':
                laystr = ', '.join(data.layer.unique())
                print ('WARNING: not sure if all layers loaded\n'
                       '         Found: %s' % laystr)
            else:
                try:
                    data = data[data.layer==layers]
                except:
                    print 'layers %s not found, reclassifying' % layers
                    return None
        else:
            try:
                data = data[data.layer.isin(layers)]
            except:
                print 'layers %s not found, reclassifying' % layers
                return None
    return data

def save(data, pref='', dataset='', subset=None, suffix='',
         savedata=True, ext='pkl', **kwargs):
    name = '_'.join(filter(None, [pref, dataset, subset, suffix]))
    name += '.' + ext
    if savedata:
        pickle.dump(data, open(name, 'wb'))
        print 'saved to', name

def show(pref='', dataset='', subset=None, suffix='',
         savefig='', **kwargs):
    name = '_'.join(filter(None, ['plot', pref, dataset, subset, suffix]))
    name += '.' + savefig
    if savefig != '':
        sns.plt.savefig(name, dpi=300)
        print 'saved to', name
    else:
        sns.plt.show()

def lin_models(ModelClass, models, kind='lin', model_name=None,
            layers=None, **kwargs):

    df = []
    for model in models:
        print model,
        spl = model.split()
        if len(spl) == 2:
            model_name, layer = spl
        else:
            model_name = model
            layer = None
        m = ModelClass(model_name=model_name, layers=layer, **kwargs)
        if kind == 'lin':
            m.linear_clf()
            out = m.lin
        elif kind == 'group':
            m.dis_group(plot=False)
            out = m.group
        elif kind == 'group_diff':
            m.dis_group_diff(plot=False)
            out = m.diff

        for col in out:
            if out[col].dtypes.name == 'category':
                out[col] = out[col].astype(str)
        if layer is not None:
            out = out.loc[out.layer==layer,:]
        out.layer = model_name
        df.append(out)
    lin = pandas.concat(df, ignore_index=True)#keys=models)
    lin = lin.rename(columns={'layer':'model2'})
    lin = stats.factorize(lin)
    return lin

def corr_models(ModelClass, models1, models2,
                model_name=None, layers=None, bootstrap=True, **kwargs):

    df = []
    for model1 in models1:
        print model1,
        spl = model1.split()
        if len(spl) == 2:
            model_name1, layer1 = spl
        else:
            model_name1 = model1
            layer1 = None
        # if model1 == 'gaborjet':
        #     layers = ['magnitudes']

        m1 = ModelClass(model_name=model_name1, layers=layer1, **kwargs)
        m1.dissimilarity()
        df_filt = m1.filter_imagenet()
        if df_filt is not None:
            sel = np.array(df_filt.kind != 'unknown')
        else:
            sel = None

        for model2 in models2:
            print model2

            spl = model2.split()
            if len(spl) == 2:
                model_name2, layer2 = spl
            else:
                model_name2 = model2
                if len(models2) == 1:
                    layer2 = 'all'
                else:
                    layer2 = None

            # if model2 == 'gaborjet':
            #     layers = ['magnitudes']
            m2 = ModelClass(model_name=model_name2, layers=layer2, **kwargs)
            m2.dissimilarity()
            d1 = m1.dis.values()[0]
            d1[np.diag_indices(len(d1))] = np.nan
            if sel is not None:
                d1 = d1[sel][:,sel]

            inds = np.triu_indices(d1.shape[0], k=1)
            if layer2 == 'all':
                for n2, (l2, d2) in enumerate(m2.dis.items()):
                    dims = m2.dims[model1].ravel()
                    d2[np.diag_indices(len(d2))] = np.nan
                    if sel is not None:
                        d2 = d2[sel][:,sel]
                        dims = dims[sel]
                    c = corr(d1, d2)
                    if bootstrap:
                        ci = stats.bootstrap_matrix(d1, d2, struct=dims)
                    else:
                        ci = [np.nan, np.nan]
                    df.append([model1, l2, c, ci[0], ci[1]])

                    # if n2 > 0:
                    #     c = partial_corr(d1[inds], d2[inds], dz2[inds])
                    #     df.append([model1, l2, c])
                    # dz2 = d2

            else:
                dims = m2.dims[model1].ravel()
                d2 = m2.dis.values()[0]
                d2[np.diag_indices(len(d2))] = np.nan
                if sel is not None:
                    d2 = d2[sel][:,sel]
                    dims = dims[sel]
                c = corr(d1, d2)
                # import pdb; pdb.set_trace()

                # ci = stats.bootstrap_permutation(d1, d2, corr)
                if bootstrap:
                    ci = stats.bootstrap_matrix(d1, d2, struct=dims)
                else:
                    ci = [np.nan, np.nan]
                # ci = boot.ci(np.dstack((d1, d2)),
                #              statfunction=corr_boot, method='pi')
                # import pdb; pdb.set_trace()
                df.append([model_name1, model_name2, c, ci[0], ci[1]])

    # df = pandas.concat(df, ignore_index=True)
    df = pandas.DataFrame(df, columns=['model1', 'model2',
                          'correlation', 'ci_low', 'ci_high'])
    # df = stats.factorize(df)
    return df

def corr(data1, data2):
    inds = np.triu_indices(data1.shape[0], k=1)
    d1 = (data1 + data1.T) / 2.
    d2 = (data2 + data2.T) / 2.
    c = np.corrcoef(d1[inds], d2[inds])[0,1]
    return c

def partial_corr(x, y, z):
    x = np.mat(x)
    y = np.mat(y)
    z = np.mat(z)

    beta_x = scipy.linalg.lstsq(z.T, x.T)[0]
    beta_y = scipy.linalg.lstsq(z.T, y.T)[0]

    res_x = x.T - z.T.dot(beta_x)
    res_y = y.T - z.T.dot(beta_y)

    pcorr = np.corrcoef(res_x.T, res_y.T)[0,1]
    return pcorr

def corr_old(data1, data2):

    def func(d1, d2):
        inds = np.triu_indices(d1.shape[0], k=1)
        c = np.corrcoef(d1[inds], d2[inds])[0,1]
        return c

    df = []
    for l1, d1 in data1.items():
        for l2, d2 in data2.items():
            c = func(d1, d2)
            # ci = stats.bootstrap_permutation(d1, d2, func)
            # df.append([l1, l2, c, ci[0], ci[1]])
            df.append([l1, l2, c])

    df = pandas.DataFrame(df, columns=['model1', 'model2', 'correlation'])
                                    #   'conf_low', 'conf_high'])
    return df

def corr_single_model(model_name, models, **kwargs):

    # models2 = ['Places205-CNN conv%d' % i for i in range(1,6)] + \
    #           ['Places205-CNN fc%d' % i for i in range(6,9)]
    df = corr_models([model_name], models1, **kwargs)

    # if model_name == 'GoogleNet':
    #     sel = ['pool_proj' in l and 'inception' in l or 'inception' not in l for l in df['GoogleNet layer']]
    #     df = df[sel]
    #     # df = df.reset_index()
    #     # del df['index']

    # if model_name == 'GoogleNet':
    #     df = df.rename(columns={'model1': 'model', 'model2': '%s layers' % model_name})
    #     g = sns.factorplot('%s layers' % model_name, 'correlation', 'model', data=df,
    #                       kind='point', markers='None', legend=False)
    #     g.axes.flat[0].set_xticklabels([])

    #     import matplotlib.lines as mlines
    #     handles = []
    #     for mname, color in zip(models1, colors):
    #         patch = mlines.Line2D([], [], color=color, label=mname)
    #         handles.append(patch)
    #     g.axes.flat[0].legend(handles=handles, loc='best')
    # else:
    df = df.rename(columns={'model1': 'model', 'model2': '%s layer' % model_name})
    print df
    import pdb; pdb.set_trace()

    plot_single_model(df, model_name)
    colors = sns.color_palette('Set2', 8)[:len(models1)]
    plot_ci(df)

    #g.axes.flat[0].set_ylim([-.1, 1])
    # chances = [1./len(np.unique(val)) for val in Stefania().dims.values()]
    # for ax, chance in zip(g.axes.flat, chances):
    #     ax.axhline(chance, ls='--', c='.2')
    show(pref='corr', suffix=model_name, **kwargs)

def compare_all(ModelClass, kind='corr', subplots=True, **kwargs):
    models = SHALLOW + DEEP #+ HUMAN
    dims = ModelClass(**kwargs).dims
    if kind == 'corr':
        try:
            del dims['natural-vs-manmade']
        except:
            pass
        df = corr_models(ModelClass, dims.keys(), models, **kwargs)
        values = 'correlation'
    elif kind in ['lin', 'group']:
        df = lin_models(ModelClass, models, kind=kind, **kwargs)
        values = 'accuracy'
    elif kind == 'group_diff':
        df = lin_models(ModelClass, models, kind=kind, **kwargs)
        values = 'preference for shape'
    df = df.rename(columns={'model1': 'kind', 'model2': 'models'})

    # base.save(df, pref='corr', suffix='CaffeNet', **kwargs)
    plot_all(df, dims, values=values, subplots=subplots,
             pref=kind, **kwargs)
    print df

def plot_all(df, dims, values='correlation', subplots=True,
             pref='', colors=None, ylim=[-.1, 1.1], **kwargs):
    if colors is None:
        colors = sns.color_palette('Set2', 8)
    # if subplots:
    if 'kind' in df:
        for i, kind in enumerate(df.kind.unique()):
            _plot_all(df[df.kind==kind], colors[i], kind, dims[kind], subplots=subplots, pref=pref, values=values, ylim=ylim, **kwargs)
    else:
        _plot_all(df, colors[1], kind='', value=None, subplots=True, pref=pref, values=values, ylim=ylim, **kwargs)
    # else:
    #     _plot_all(df, subplots=subplots)

def _plot_all(df, color, kind='', value=None, subplots=True, pref='',
              values='correlation', ylim=[-.1, 1.1], **kwargs):
    colors = sns.color_palette('Set2', 8)
    gray = colors[-1]
    palette = [gray]*len(SHALLOW) + [color]*len(DEEP)
    if subplots:
        hue = None
    else:
        hue = 'kind'
    g = sns.factorplot('models', values, hue=hue,
                        data=df, kind='bar', palette=palette)

    sns.plt.ylim(ylim)
    #import pdb; pdb.set_trace()

    # for r, (rowno, row) in enumerate(df.iterrows()):
    #     g.axes.flat[0].plot([r,r], [row.ci_low, row.ci_high],
    #                         color='.15',
    #                         lw=sns.mpl.rcParams["lines.linewidth"] * 1.8)
    if 'ci_low' in df.columns:
        plot_ci(df)
    if 'accuracy' in df.columns and value is not None:
        plot_chance(value)



    # kinds = df.kind.unique()
    # palette = [[gray]*len(SHALLOW) + [color]*len(DEEP) for color in pal[:len(kinds)]]
    # recolor(g, palette=palette, what=['Rectangle'])
    set_vertical_labels(g)
    # for ax, kind in zip(g.axes.flat, kinds):
    sns.plt.title(kind)

    # chances = [1./len(np.unique(val)) for val in Stefania().dims.values()]
    # for ax, chance in zip(g.axes.flat, chances):
    #     ax.axhline(chance, ls='--', c='.2')
    if pref != '':
        show(pref=pref + '_' + kind, suffix='all', **kwargs)
    else:
        show(pref=kind, suffix='all', **kwargs)

def corr_all_layers(ModelClass, kinds=['shape', 'category'], colors=None,
                    model_name=None, layers=None, ylim=[0,1], **kwargs):
    if colors is None:
        colors = sns.color_palette('Set2')[:len(kinds)]
    fig, axes = sns.plt.subplots(len(DEEP), sharey=True, figsize=(2.5,4))
    for model, ax in zip(DEEP, axes):
        corr = corr_models(ModelClass, kinds, [model], bootstrap=False, **kwargs)
        for kind, color in zip(kinds, colors):
            sel = corr[corr.model1==kind]
            ax.plot(range(len(sel)), np.array(sel.correlation), lw=3, color=color)
            ax.set_xlim([0, len(sel)-1])
        ax.set_xticklabels([])
        ax.set_title(model)
        sns.despine()

    sns.plt.ylim(ylim)
    ax.set_yticks([0,.5,1])
    ax.set_yticklabels(['0','.5','1'])
    show(pref='corr', suffix='all_layers', **kwargs)

def lin_all(ModelClass, subplots=True, **kwargs):
    models = SHALLOW + DEEP
    dims = ModelClass(**kwargs).dims
    df = lin_models(ModelClass, dims.keys(), models, **kwargs)
    df = df.rename(columns={'model1': 'kind', 'model2': 'models'})
    # base.save(df, pref='corr', suffix='CaffeNet', **kwargs)
    plot_all(df, dims, values='accuracy', subplots=subplots,
             pref='lin', **kwargs)
    print df

    # shallow = ['px', 'gaborjet', 'hog', 'phog', 'phow']
    # models = shallow + deep
    # lin = lin_models(ModelClass, models, **kwargs)
    # lin = lin[lin.kind==kind]

    # gray = sns.color_palette('Set2', 8)[-1]
    # orange = sns.color_palette('Set2', 8)[1]
    # palette = [gray] * len(shallow) + [orange] * len(deep)

    # g = sns.factorplot('model', 'accuracy', data=lin,
    #                     kind='bar', palette=palette)
    # chance = 1. / len(lin.actual.unique())
    # g.axes.flat[0].axhline(chance, ls='--', c='.2')
    # set_vertical_labels(g)
    # show(pref='lin', **kwargs)

def set_vertical_labels(g):
    for ax in g.axes.flat:
        for n, model in enumerate(ax.get_xticklabels()):
            ax.text(n, .03, model.get_text(), rotation='vertical',
                    ha='center', va='bottom', backgroundcolor=(1,1,1,.5))
        ax.set_xticklabels([])
