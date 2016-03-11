from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, glob, urllib, subprocess
from collections import OrderedDict

import numpy as np
import scipy
import pandas
from nltk.corpus import wordnet as wn
import seaborn as sns
import statsmodels.formula.api as smf

sys.path.insert(0, '../../psychopy_ext')
from psychopy_ext import models, stats, plot

ORDER = ['color', 'gray', 'silhouette']
import base


class Snodgrass(base.Base):

    def __init__(self, *args, **kwargs):
        kwargs['skip_hmo'] = True
        super(Snodgrass, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.dims = OrderedDict([('shape', np.repeat(range(6),6))])
        self.colors = OrderedDict([('shape', base.COLORS[1])])

    def get_images(self):
        url = 'https://ndownloader.figshare.com/articles/3102781/versions/1'
        path = 'snodgrass/img/orig'
        if not os.path.isdir(path): os.makedirs(path)
        fnames = super(Snodgrass, self).download_dataset(url, path=path, ext='.png')
        # for fname in fnames:
        #     if os.path.splitext(fname)[1] == '.PCT':
        #         new_path = os.path.splitext(fname)[0][:-1] + '.PCT'
        #         os.rename(fname, new_path)
        self._create_stim()

    def set_models(self):
        if self.skip_hmo:
            deep = [d for d in base.DEEP if d!='hmo']
        else:
            deep = base.DEEP
        self.models = [('deep',m) for m in deep]

    @base.get_data('preds')
    def predict(self):
        _, sel = self.filter_synset_ids()
        try:
            m = models.get_model(self.model_name)
        except:
            base.msg('%s is not available for generating responses' %self.model_name)
            raise Exception
        else:
            m.load_image = base.load_image
            preds = m.predict(self.ims, topn=1000)
            # limit to top 5 guesses that are in Snodgrass
            top5 = []
            for pred in preds:
                tmp = []
                for p in pred:
                    if p['synset'] in sel: tmp.append(p)
                    if len(tmp) == 5:
                        top5.append(tmp)
                        break
            self.save(top5, 'preds')
        return preds

    def determine_synonyms(self):
        imagenet_labels = self.synsets_from_txt('synset_words.txt')
        df = pandas.read_csv(os.path.join(self.exp, 'data', self.exp + '.csv'), sep='\t')
        df.loc[df.synonyms.isnull(), 'synonyms'] = ''
        df.loc[df.synonyms_extra.isnull(), 'synonyms_extra'] = ''
        outf = open('snodgrass_syns.csv', 'ab')
        for idx, dtlab in df.iterrows():
            print('===  {}: {}  ==='.format(idx, dtlab['name']))
            vals = map(str, dtlab[['no','synset_id','name','description']].values)
            words = dtlab.synonyms.split(', ')
            extra = dtlab.synonyms_extra.split(', ')
            if len(extra[0]) > 0: works += extra
            for word in words:
                print('### {}:'.format(word))
                synsets = wn.synsets(word, pos=wn.NOUN)
                if len(synsets) == 0:
                    outf.write('\t'.join(vals + [word, '']) + '\n')
                else:
                    for i,s in enumerate(synsets):
                        print('{} - {}'.format(i, s.definition()))
                    choices = raw_input('Choice(s): ')
                    if choices == '':
                        outf.write('\t'.join(vals + [word, '']) + '\n')
                    else:
                        choices = map(int, choices.split(','))
                        for c in choices:
                            synid = synsets[c].pos() + '{:0>8d}'.format(synsets[c].offset())
                            outf.write('\t'.join(vals + [word, synid]) + '\n')
                print()
            print()
        outf.close()

    def filter_synset_ids(self):
        imagenet_labels = self.synsets_from_txt('synset_words.txt')
        imagenet_ids = [iml['id'] for iml in imagenet_labels]
        df = pandas.read_csv(os.path.join(self.exp, 'data', self.exp + '_syns.csv'), sep='\t')
        # allowed_ids = np.concatenate([df.synset_id, df.synonym_id])
        imagenet_ids_f = [i for i in imagenet_ids if i in df.synset_id.values]
        snodgrass_ids_f = [i for i in imagenet_ids if i in df.synset_id.values]
        return snodgrass_ids_f, imagenet_ids_f

    def pred_acc(self, compute_acc=True):
        # if compute_acc:
        preds = self.predict()
        sn_ids, img_ids = self.filter_synset_ids()
        # imagenet_labels = self.synsets_from_txt('synset_words.txt')
        # imagenet_ids = [iml['id'] for iml in imagenet_labels]
        # dataset_labels = self.synsets_from_csv(os.path.join(self.exp, 'data', self.exp + '.csv'))
        # all_hyps = lambda s:s.hyponyms()

        # df = pandas.DataFrame.from_dict(dataset_labels)
        df = pandas.read_csv(os.path.join(self.exp, 'data', self.exp + '_syns.csv'), sep='\t')
        df.loc[df.synonym.isnull(), 'synonym'] = ''
        df.loc[df.synonym_id.isnull(), 'synonym_id'] = ''

        f = lambda x: ','.join([i for i in x if len(i) > 0])
        agg = df.groupby(['no', 'synset_id', 'name', 'description']).aggregate(f).reset_index()

        agg['corr_resp_ids'] = ''
        agg['sel'] = np.nan
        agg['model_resp_id'] = ''
        agg['model_resp_name'] = ''
        agg['model_confidence'] = np.nan
        agg['model_accuracy'] = np.nan

        for (idx, row), pred in zip(agg.iterrows(), preds):
            ids = [row.synset_id] #+ row.synonym_id.split(',')
            agg.loc[idx, 'corr_resp_ids'] = ','.join(ids)
            agg.loc[idx, 'sel'] = row.synset_id in sn_ids #any([i in imagenet_ids for i in ids])
            agg.loc[idx, 'model_resp_id'] = pred[0]['synset']
            agg.loc[idx, 'model_resp_name'] = pred[0]['label']
            agg.loc[idx, 'model_confidence'] = pred[0]['confidence']
            agg.loc[idx, 'model_accuracy'] = agg.loc[idx, 'model_resp_id'] in ids



            # synset = wn._synset_from_pos_and_offset(dtlab.synset_id[0],
            #                                         int(dtlab.synset_id[1:]))
            # hypos = set([i for i in synset.closure(all_hyps)])
            # hypos = hypos.union([dtlab['synset']])
            # for imglab in imagenet_labels:
            #     if imglab['synset'] in hypos:
            #         df.loc[no, 'imgid'] = imglab['id']
            #         df.loc[no, 'imgnames'] = imglab['names']
            #         if imglab['id'] == df.loc[no, 'id']:
            #             df.loc[no, 'kind'] = 'exact'
            #         else:
            #             df.loc[no, 'kind'] = 'superordinate'
            #         break
            # if compute_acc:
            #     acc = False
            #     acc0 = False
            #     for i,p in enumerate(preds[no]):
            #         psyn = wn._synset_from_pos_and_offset(p['synset'][0],
            #                                               int(p['synset'][1:]))
            #         df.loc[no, 'pred%d'%i] = ', '.join(psyn.lemma_names())
            #         # check if the prediction is exact
            #         # or at least more specific than the correct resp
            #         if psyn in hypos:
            #             acc = True
            #         if i==0:
            #             if psyn in hypos:
            #                 acc0 = True
            #     if acc == False:
            #         if df.loc[no, 'kind'] != 'unknown':
            #             df.loc[no, 'accuracy'] = False
            #     else:
            #         df.loc[no, 'accuracy'] = True
            #     if acc0 == False:
            #         if df.loc[no, 'kind'] != 'unknown':
            #             df.loc[no, 'accuracy0'] = False
            #     else:
            #         df.loc[no, 'accuracy0'] = True
            #     df.loc[no, 'confidence0'] = preds[no][0]['confidence']
        # import pdb; pdb.set_trace()
        return agg

    def acc_single(self):
        df = self.pred_acc()
        # acc_exact = df[df.kind=='exact'].accuracy0.mean()
        # print('Exact match: {:.2f}'.format(acc_exact))
        # print('Exact match or more specific: {:.2f}'.format(df.accuracy0.mean()))
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

    def synsets_from_csv(self, fname):
        sf = pandas.read_csv(fname, sep='\t')
        df = []
        for idx, row in sf.iterrows():
            idd = row['synset_id']
            try:
                synset = wn._synset_from_pos_and_offset(idd[0], int(idd[1:]))
            except:
                import pdb; pdb.set_trace()

            df.append({'id':idd, 'names':row['name'], 'synset':synset})
        return df

    def pred_corr(self, value='accuracy', method='corr'):
        human = base.load(pref='preds', exp=self.exp, suffix='human')
        human = human.groupby(['kind', 'no']).acc.mean()

        dfs = []
        for subset in ORDER:
            self.set_subset(subset)
            df = self._pred_corr(human.loc[subset], value=value, method=method)
            df['dataset'] = subset
            dfs.append(df)
        df = pandas.concat(dfs, ignore_index=True)
        print(df.groupby('dataset').mean())
        if self.task == 'run':
            self.plot_single(df, 'pred_corr')
        return df

    def _pred_corr(self, human, value='accuracy', method='corr'):
        nname = models.NICE_NAMES[self.model_name].lower()
        acc = self.acc_single()
        acc[value] = acc[value].astype(np.float)
        # import pdb; pdb.set_trace()
        # sel = acc.kind!='unknown'
        sel = acc.sel.copy()
        acc = acc[sel][value]
        sns.set_palette(sns.color_palette('Set2')[1:])

        df = []
        human = human[sel.values]
        if method == 'corr':
            f = lambda machine, human: (1 + stats.corr(machine, human)) / 2.
        elif method == 'diff':
            f = lambda machine, human: 1 - np.mean(np.abs(machine-human))
        elif method == 'euclidean':
            f = lambda machine, human: 1 - scipy.spatial.distance.sqeuclidean(machine, human) / len(machine)
        else:
            raise Exception('Method {} not recognized'.format(method))

        corr = f(acc, human)
        if self.bootstrap:
            print('bootstrapping stats...')
            bf = stats.bootstrap_resample(acc, human, func=f, ci=None, seed=0)
            c = np.vstack([np.repeat(corr, len(bf)), np.arange(len(bf)), bf])
            df.extend(c.T.tolist())
        else:
            df.append([corr, 0, np.nan])
        df = pandas.DataFrame(df, columns=['consistency', 'iter', 'bootstrap'])
        # self.save(df, pref='pred_corr')
        return df

    def corr(self):
        self.set_subset('silhouette')
        df = self.acc_single()
        df['dataset'] = 'silhouette'
        human = base.load(pref='preds', exp=self.exp, suffix='human')
        df['human_accuracy'] = np.nan
        n = len(df.dataset.unique())
        mean_acc = human[human.kind=='silhouette'].groupby('no').acc.mean()
        df.loc[:,'human_accuracy'] = mean_acc.tolist() * n
        df = df[df.sel]
        df.model_accuracy = df.model_accuracy.astype(int)
        sns.set_palette(sns.color_palette('Set2')[1:])
        self._corr(df, 'all')
        # self._corr(df[df.kind=='exact'], 'exact')

    def corr_olddata(self):
        self.set_subset('silhouette')
        df = self.acc_single()
        df['dataset'] = 'silhouette'
        human = pandas.read_csv('snodgrass/data/sil_human_acc.csv',header=None)
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
        formula = str('model_accuracy ~ human_accuracy')
        logreg = smf.logit(formula=formula, data=sel).fit()
        summ = logreg.summary()
        if self.html is None:
            print(summ)
        else:
            summ = summ.as_html().replace('class="simpletable"',
                                          'class="simpletable table"')

        sel = sel.rename(columns={'human_accuracy': 'human accuracy',
                                  'model_accuracy': 'model accuracy'})

        sns.lmplot('human accuracy', 'model accuracy', data=sel, x_jitter=.01,
                    y_jitter=.05, logistic=True, truncate=True)

        bins = np.digitize(sel['human accuracy'], np.arange(.05,1,.1))
        #bins[bins==11] = 10
        count = sel['model accuracy'].groupby(bins).count()
        mean = sel['model accuracy'].groupby(bins).mean()
        sns.plt.scatter(.1*mean.index, mean, s=10*count, c='.15',
                        linewidths=0, alpha=.8)
        sns.plt.title(models.NICE_NAMES[self.model_name])
        sns.plt.xlim([-.1, 1.1])
        sns.plt.ylim([-.1, 1.1])
        self.show(pref='corr_sil', suffix=self.model_name + '_' + suffix,
                  caption=suffix + summ)

    def behav(self):
        self.model_name = 'behav'
        human = base.load(pref='preds', exp=self.exp, suffix='human')
        sil = pandas.read_csv('snodgrass/data/sil_human_acc.csv', header=None) / 100.
        sil2 = human.groupby(['kind', 'no']).acc.mean()
        corr = np.corrcoef(sil.values.ravel(), sil2['silhouette'].values)[0,1]
        cons = 1 - scipy.spatial.distance.sqeuclidean(sil.values.ravel(),
                                            sil2['silhouette'].values) / 260
        sel, _ = self.filter_synset_ids()
        human = human[human.synset_id.isin(sel)]

        if self.task == 'run':
            sns.factorplot('kind', 'acc', data=human, units='subjid',
                            kind='bar', color=self.colors['shape'])
            self.show(pref='acc')
            self.html.writetable(human.groupby('kind').acc.mean())
            self.html.write('<p>Correlation old-new: {:.2f}</p>'.format(corr))
            self.html.write('<p>Consistency old-new: {:.2f}</p>'.format(cons))
        return human

    def behav_olddata(self):
        self.model_name = 'behav'
        sil = pandas.read_csv('snodgrass/data/sil_human_acc.csv', header=None) / 100.
        df = [('color', .903, .169),
              ('gray', .892, .172),
            #   ('line drawing', .882, .171),
              ('silhouette', sil.mean().values[0], sil.std(ddof=1).values[0])]
        df = pandas.DataFrame(df, columns=['dataset', 'accuracy', 'stdev'])
        n = 260
        ci = df.stdev * 1.96 / np.sqrt(n)
        df['ci_low'] = df.accuracy - ci
        df['ci_high'] = df.accuracy + ci

        sns.factorplot('dataset', 'accuracy', data=df,
                        kind='bar', color=self.colors['shape'])
        hue = 'kind' if 'kind' in df else None
        plot.plot_ci(df, what=['Rectangle'], hue=hue)
        self.show(pref='acc')
        self.html.writetable(df)

    def _create_stim(self):
        for s in ORDER:
            path = os.path.join('snodgrass', 'img', s)
            if not os.path.isdir(path): os.makedirs(path)

        # gen2nd = GenSecondOrder()
        # subprocess.call('mogrify -format png snodgrass/img/orig/*.PCT'.split())
        for fn in sorted(glob.glob('snodgrass/img/orig/*.png')):
            print('\r{}'.format(os.path.basename(fn)), end='')
            sys.stdout.flush()
            # im = scipy.misc.imread(fn)
            # try:
            #     if max(im.shape) != 281:
            #         scale = 281. / max(im.shape)
            #         rim = scipy.misc.imresize(im, scale)
            #         scipy.misc.imsave(fn, rim)
            # except:
            #     import pdb; pdb.set_trace()

            newname = os.path.basename(fn)

            # square color images
            cname = os.path.join('snodgrass', 'img', ORDER[0], newname)
            # RGBA version
            # subprocess.call('convert {0} -alpha set -channel RGBA -fuzz 10% '
            #                 '-fill none -floodfill +0+0 white '
            #                 '-gravity center -background none '
            #                 '-extent 281x281 {1}'.format(fn, cname).split())
            # RGB version
            subprocess.call('convert {} -gravity center -resize 256x256 -background white '
                            '-extent 256x256 {}'.format(fn, cname).split())

            # grayscaled
            newfname = os.path.join('snodgrass', 'img', ORDER[1], newname)
            subprocess.call('convert {} -colorspace '
                            'gray {}'.format(cname, newfname).split())
            # -channel RGBA -matte

            # line drawings
            # newfname = os.path.join('snodgrass', 'img', suffix[2], newname)
            # subprocess.call('convert {} -negate -separate -lat 5x5+5% -negate '
            #                 '-evaluate-sequence add {}'.format(cname, newfname).split())

            # silhouettes
            newfname = os.path.join('snodgrass', 'img', ORDER[2], newname)
            subprocess.call('convert {} -alpha set -channel RGBA -fuzz 10% '
                            '-fill none -floodfill +0+0 white '

                            '-blur 1x1 -alpha extract '
                            '-negate {}'.format(cname, newfname).split())

            # 2nd order edges
            # im = imread(newfname, flatten=True)
            # mask = np.zeros(im.shape).astype(bool)
            # mask[im==0] = True
            # newfname = os.path.join('snodgrass', 'img', suffix[4], newname)
            # gen2nd.gen(mask=im, savename=newfname)
        print('\r       ')

    def pdf2syns(self):
        with open('snodgrass/data/sn_list.txt') as f: tmp = f.readlines()
        sn_list = []
        for s in tmp:
            spl = s.split('. ')
            sn_list.append((spl[0], spl[1].strip('\r\n')))

        with open('.snodgrass/data/sn_syns.txt') as f: tmp = f.readlines()
        sn_syns = []
        line = ''
        upper = list(map(unicode, map(chr, range(65, 91))))
        for s in tmp:
            s = s.strip('\r\n')
            if unicode(s) == u'-':
                sn_syns.append(line)
                line = ''
            elif s[0] in upper or s=='12-inch ruler':
                sn_syns.append(line)
                line = s
            else:
                line += ' ' + s
        sn_syns.append(line)
        sn_syns = sn_syns[1:]

        df = {}
        for w, s in zip(sn_list, sn_syns):
            clean = []
            for t in s.split(','):
                t = t.lstrip()
                spl = t.split()
                if len(spl) > 0:
                    if spl[-1][0] in '0123456789': spl = spl[:-1]
                if len(spl) != 0:
                    spl = ' '.join(spl)
                    if spl not in ['Lincoln', 'TV']:
                        spl = spl.lower()
                    clean.append(spl)
            if len(clean) == 0:
                df[int(w[0])] = [w[1].lower(), '']
            else:
                df[int(w[0])] = [w[1].lower(), ', '.join(clean)]

        with open('snodgrass/data/snodgrass.csv.orig') as f: tmp = f.readlines()
        f = open('snodgrass/data/snodgrass.csv', 'wb')

        f.write('no\tsynset_id\tname\tdescription\tsynonyms\tsynonyms_extra\n')
        extra = {13: 'stroller, pram',
                 77: 'doorhandle',
                 185: 'fridge',
                 182: 'bunny, hare'}

        for ln, line in enumerate(tmp):
            spl = line.split(',')
            n = spl[0]
            term = spl[1]
            descr = ', '.join(spl[2:]).strip('\r\n')
            if ln+1 in df:
                if df[ln+1][0] != term:
                    print(df[ln+1][0], term)
                syns = df[ln+1][1]
            else:
                syns = ''
            out = [str(ln+1), n, term, descr, syns]
            out += [extra[ln+1]] if ln+1 in extra else ['']
            f.write('\t'.join(out) + '\n')

        f.close()

class Compare(base.Compare):
    def __init__(self, *args):
        super(Compare, self).__init__(*args)

    def classify(self, clear_memory=False):
        for depth, model_name in self.myexp.models:
            self.myexp.set_model(model_name)
            for subset in ORDER:
                self.myexp.set_subset(subset)
                self.myexp.classify()
                if clear_memory: del self.myexp.resps

    def dissimilarity(self):
        for depth, model_name in self.myexp.models:
            self.myexp.set_model(model_name)
            for subset in ORDER:
                self.myexp.set_subset(subset)
                self.myexp.dissimilarity()

    def predict(self):
        for depth, model_name in self.myexp.models:
            if depth == 'deep':
                self.myexp.set_model(model_name)
                for subset in ORDER:
                    self.myexp.set_subset(subset)
                    self.myexp.predict()

    def accuracy(self):
        df = self._acc()
        df = df[df.sel]
        sns.factorplot(x='dataset', y='model_accuracy', hue='model', data=df,
                        kind='bar', color=self.myexp.colors['shape'])
        sns.plt.ylim([0,1])
        self._plot_behav()
        base.show(pref='acc', exp=self.myexp.exp, suffix='all_acc', savefig=self.myexp.savefig, html=self.myexp.html)
        return df

    def corr_models(self):
        df = self._acc()
        df = df[df.sel]
        corr = lambda x,y: 1 - scipy.spatial.distance.sqeuclidean(x,y) / len(x)

        cr = []
        for m1 in df.model.unique():
            d1 = df[df.model==m1]
            for s1 in df.dataset.unique():
                for m2 in df.model.unique():
                    d2 = df[df.model==m2]
                    for s2 in df.dataset.unique():
                        if m1==m2 and s1==s2:
                            r = np.nan
                        else:
                            r = corr(d1[d1.dataset==s1].model_accuracy.values,
                                     d2[d2.dataset==s2].model_accuracy.values)
                        cr.append([m1,s1,m2,s2,r])
        cr = pandas.DataFrame(cr, columns=['model1', 'dataset1',
                              'model2', 'dataset2', 'lr'])
        cr = stats.factorize(cr)
        crs = cr.set_index(['model1', 'dataset1', 'model2', 'dataset2'])
        crs = crs.unstack(['model2', 'dataset2'])
        sns.plt.figure()
        sns.heatmap(crs)
        print(crs)
        print(cr.groupby(['dataset1', 'dataset2']).mean())
        self.show(pref='mcorr')

        if self.myexp.html is not None:
            self.myexp.html.writetable(crs, caption='proportion of matching responses')
            g = cr.groupby(['dataset1', 'dataset2']).mean()
            self.myexp.html.writetable(g, caption='mean proportion of matching responses')

    def _acc(self):
        dfs = []
        for depth, model_name in self.myexp.models:
            if depth == 'deep':
                self.myexp.set_model(model_name)
                df = self.myexp.accuracy()
                dfs.append(df)
        df = pandas.concat(dfs, ignore_index=True)
        return df

    def _plot_behav(self):
        human = self.myexp.behav()
        hacc = human.groupby(['kind', 'subjid']).acc.mean()
        for sno, subset in enumerate(ORDER):
            floor, ceiling = stats.bootstrap_resample(hacc[subset])
            # floor = np.percentile(hacc[subset], 2.5)
            # ceiling = np.percentile(hacc[subset], 97.5)
            sns.plt.axhspan(floor, ceiling, xmin=(2*sno+1)/6.-1/7., xmax=(2*sno+1)/6.+1/7., facecolor='0.9', edgecolor='0.9', zorder=0)
            sns.plt.axhline(hacc[subset].mean(), xmin=(2*sno+1)/6.-1/7., xmax=(2*sno+1)/6.+1/7., color='0.65')

    def pred_corr(self):
        pref = 'pred_corr'
        print()
        print('{:=^50}'.format(' ' + pref + ' '))
        value = 'model_accuracy'
        method = 'euclidean'
        df = self.get_data_all(pref, kind='compare', value=value, method=method)

        behav = self.myexp.behav()
        behav = behav.pivot_table(index=['kind', 'subjid'],
                                  columns='no', values='acc')

        df = base._set_ci(df, groupby=['models', 'dataset'])
        g = sns.factorplot(x='dataset', y='consistency', hue='models',
                            data=df, kind='bar',
                            color=self.myexp.colors['shape'])
        hue = 'kind' if 'kind' in df else None
        plot.plot_ci(df, hue=hue)
        sns.plt.ylim([0,1])
        for sno, subset in enumerate(ORDER):
            self.myexp.set_subset(subset)
            rel = self.reliability(behav.loc[subset])
            # import pdb; pdb.set_trace()
            rel = ((1+rel[0])/2., (1+rel[1])/2.)
            sns.plt.axhspan(rel[0], rel[1], xmin=(2*sno+1)/6.-1/7., xmax=(2*sno+1)/6.+1/7., facecolor='0.9', edgecolor='0.9', zorder=0)
        self.show(pref=pref, suffix='all_' + value + '_' + method)

    def reliability(self, data):
        """
        Computes upper and lower boundaries of data reliability

        :Args:
            data (np.ndarray)
                N samples x M features
        :Returns:
            (floor, ceiling)
        """
        corr = lambda x,y: 1 - scipy.spatial.distance.sqeuclidean(x,y) / len(x)
        zdata = np.array(data)
        zmn = np.mean(zdata, axis=0)
        ceil = np.mean([corr(subj,zmn) for subj in zdata])
        rng = np.arange(zdata.shape[0])

        floor = []
        for s, subj in enumerate(zdata):
            mn = np.mean(zdata[rng!=s], axis=0)
            floor.append(corr(subj,mn))
        floor = np.mean(floor)
        return floor, ceil


def report(**kwargs):
    html = kwargs['html']
    html.writeh('Snodgrass', h='h1')

    html.writeh('Accuracy', h='h2')

    html.writeh('Old behavioral', h='h3')
    kwargs['layers'] = 'probs'
    kwargs['task'] = 'run'
    kwargs['func'] = 'behav_olddata'
    myexp = Snodgrass(**kwargs)
    myexp.behav_olddata()

    html.writeh('New behavioral', h='h3')
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
    html.writeh('Per model', h='h3')
    kwargs['layers'] = 'probs'
    kwargs['task'] = 'run'
    kwargs['func'] = 'corr'
    myexp = Snodgrass(**kwargs)
    for depth, model_name in myexp.models:
        if depth == 'deep':
            myexp.set_model(model_name)
            myexp.corr()

    html.writeh('New data', h='h3')
    kwargs['layers'] = 'preds'
    kwargs['task'] = 'compare'
    kwargs['func'] = 'pred_corr'
    myexp = Snodgrass(**kwargs)
    Compare(myexp).pred_corr()

    html.writeh('Between models', h='h3')
    kwargs['layers'] = 'preds'
    kwargs['task'] = 'compare'
    kwargs['func'] = 'corr_models'
    myexp = Snodgrass(**kwargs)
    Compare(myexp).corr_models()
