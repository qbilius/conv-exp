from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import matplotlib
# matplotlib.use('Agg')

import sys, os, itertools, copy, hashlib, glob, datetime, argparse, shutil, warnings
from collections import OrderedDict
import cPickle as pickle

import numpy as np
import scipy
import pandas
import skdata
import skimage, skimage.io, skimage.transform
import seaborn as sns

import boto
import pymongo
import pypsignifit as psi

import tabular as tb
import mturkutils
import dldata.stimulus_sets.hvm

from psychopy_ext import utils, stats


MTURKLIB = os.path.abspath(os.path.join(mturkutils.__path__[0], os.pardir, 'lib')) + '/'
HVM_10 = [ 'bear', 'ELEPHANT_M', '_18', 'face0001', 'alfa155', 'breed_pug', 'TURTLE_L', 'Apple_Fruit_obj',  'f16', '_001']


class Experiment(mturkutils.base.Experiment):

    def __init__(self, single=False, short=False, save=True, *args, **kwargs):
        self.single = single
        self.short = short
        self.save = save
        super(Experiment, self).__init__(*args, **kwargs)
        if self.sandbox:
            print('**WORKING IN SANDBOX MODE**')
        delattr(self, 'meta')

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except:
            if name in ['meta','exp_plan']:
                value = getattr(self, 'get_' + name)()
                setattr(self, name, value)
                return self.__dict__[name]
            else:
                raise

    def createTrials(self):
        d = self.exp_plan

        self._trials = {
            'isi1': d['isi1'].tolist(),
            'stim': self.get_obj_url(),
            'stim_dur': d['stim_dur'].tolist(),
            'gap_dur': d['gap_dur'].tolist(),
            'mask': self.get_mask_url(),
            'mask_dur': d['mask_dur'].tolist(),
            'isi2': d['isi2'].tolist(),
            'label1': self.get_label_url('label1'),
            'label2': self.get_label_url('label2')
        }

    def get_exp_plan(self):
        # prefix = 'sandbox' if self.sandbox else 'production'
        fnames = sorted(glob.glob(self.bucket + '_exp_plan_*.pkl'))[::-1]

        if len(fnames) == 0:
            print('Creating exp_plan')
            exp_plan = self.create_exp_plan()
        else:
            if len(fnames) > 1:
                print('Multiple exp_plan files found:')
                for i, fname in enumerate(fnames):
                    print(i+1, fname, sep=' - ')
                print(0, 'Create a new exp_plan', sep=' - ')
                choice = raw_input('Choose which one to load (default is 1): ')

                if choice == '0':
                    print('Creating exp_plan')
                    self.create_exp_plan()
                else:
                    if choice == '': choice = '1'
                    try:
                        fname = fnames[int(choice)-1]
                    except:
                        raise
            else:
                fname = fnames[0]

            print('Using', fname, end='\n\n')
            exp_plan = pickle.load(open(fname))
            # exp_plan = exp_plan.addcols(np.repeat([1], len(meta)).astype(int), names=['batch'])

        if self.single:
            exp_plan = exp_plan[:self.trials_per_hit]
        elif self.short:
            exp_plan = exp_plan[:10]

        return exp_plan

    def save_exp_plan(self, exp_plan):
        # prefix = 'sandbox' if self.sandbox else 'production'
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        fname = '_'.join([self.bucket_name, 'exp_plan', date]) + '.pkl'
        pickle.dump(exp_plan, open(fname, 'wb'))
        print('Saved exp_plan to:', fname)

    def get_hitids(self):
        prefix = 'sandbox' if self.sandbox else 'production'
        pattern = self.bucket + '_' + prefix + '_hitids_*.pkl'
        fnames = sorted(glob.glob(pattern))[::-1]

        if len(fnames) == 0:
            raise Exception('No HIT ID files found with pattern ' + pattern)
        elif len(fnames) > 1:
            print('Multiple HIT ID files found:')
            for i, fname in enumerate(fnames):
                print(i+1, fname, sep=' - ')
            choice = raw_input('Choose which one to load (default is 1): ')

            if choice == '': choice = '1'
            try:
                fname = fnames[int(choice)-1]
            except:
                raise
        else:
            fname = fnames[0]

        print('Using', fname, end='\n\n')
        hitids = pickle.load(open(fname))
        return hitids

    def check_if_hits_are_completed(self):
        """
        Checks if all HITs have been completed.

        Prints the answer and also return True or False.
        """
        results = mturkutils.base.download_results(self.get_hitids(), sandbox=self.sandbox)
        worker_ids = []
        for assign, hit in results:
            if len(assign) > 0:
                worker_ids.append(assign[0].WorkerId)
                print(assign[0].WorkerId)
        print(sorted(worker_ids))  # checking for duplicates
        completed = [len(assign) > 0 for assign, hit in results]
        print('{} out of {} HITs completed'.format(sum(completed), len(completed)))
        return completed

    def data2pkl(self):
        mongo_conn = pymongo.MongoClient(host='localhost', port=22334)
        db = mongo_conn['mturk']
        coll = db[self.collection_name]
        dfs = [pandas.DataFrame(doc['ImgData']) for doc in coll.find()]
        df = pandas.concat(dfs)
        df = df[df.subjid!=11]  # this participant did it twice!
        df.to_pickle(self.bucket + '.pkl')

class SnodgrassNaming(Experiment):

    def __init__(self, **kwargs):
        self.trials_per_hit = 260
        self.nsubj = 1
        self.bucket = 'snodgrass-naming'
        max_dur = 40  # in minutes
        expected_dur = 30  # in minutes
        self.stim_s3path = 'https://s3.amazonaws.com/snodgrass-naming/'

        super(SnodgrassNaming, self).__init__(
            htmlsrc='{}-template.html'.format(self.bucket),
            htmldst='{}-n%02d.html'.format(self.bucket),
            title='Object naming',
            reward=0.50,
            duration=max_dur * 60,  # in seconds
            keywords=['neuroscience', 'psychology', 'experiment', 'object recognition'],  # noqa
            description="***You may complete ONLY ONE HIT in this group.*** ONLY FOR FLUENT ENGLISH SPEAKERS. Complete a visual object recognition task where you report the identity of objects you see. We expect this HIT to take about {expected_dur} minutes or less, though you must finish in under {max_dur} minutes. By completing this HIT, you understand that you are participating in an experiment for the Massachusetts Institute of Technology (MIT) Department of Brain and Cognitive Sciences. You may quit at any time, and you will remain anonymous. Contact the requester with questions or concerns about this experiment.".format(expected_dur=expected_dur, max_dur=max_dur),
            comment=self.bucket,
            collection_name=self.bucket,
            max_assignments=1,
            bucket_name=self.bucket,
            trials_per_hit=self.trials_per_hit,
            tmpdir='tmp',
            frame_height_pix=1200,
            othersrc=[MTURKLIB + 'dltk.js', MTURKLIB + 'dltkexpr.js', MTURKLIB +  'dltkrsvp.js', 'snodgrass_synonyms.txt'],
            additionalrules=[
                {'old': '${EXPECTED_DUR}', 'new': str(expected_dur)},
                {'old': '${MAX_DUR}', 'new': str(max_dur)},
                {'old': '${NTRIALS}', 'new': str(self.trials_per_hit)},
                {'old': '${NOBJS}', 'new': '260'}],
            **kwargs)


    def create_exp_plan(self):
        """Define each trial's parameters
        """
        df = pandas.read_csv('../data/snodgrass.csv', sep='\t')
        df['imgno'] = range(1, self.trials_per_hit+1)
        df = pandas.concat([df for i in range(self.nsubj)])
        df['subjid'] = np.repeat(range(self.nsubj), self.trials_per_hit)
        df['order'] = np.hstack([np.random.permutation(self.trials_per_hit) for i in range(self.nsubj)])
        # df['kind'] = np.repeat(['color', 'gray', 'silhouette'], self.trials_per_hit * self.nsubj // 3)
        df['kind'] = np.repeat(['color'], self.trials_per_hit * self.nsubj)
        df['isi1'] = 500
        df['stim_dur'] = 100
        df['isi2'] = 500
        df['subj_resp'] = None
        df['acc'] = np.nan
        df['rt'] = np.nan

        df = df.sort_values(by=['subjid', 'order'])
        rec = df.to_records(index=False)
        exp_plan = tb.tabarray(array=rec, dtype=rec.dtype)
        if self.save:
            self.save_exp_plan(exp_plan)
        return exp_plan

    def get_obj_url(self):
        f = lambda k,i: self.stim_s3path + 'images/{}/{:0>3d}.png'.format(k, i)
        return map(f, self.exp_plan['kind'], self.exp_plan['imgno'])

    def createTrials(self):
        d = self.exp_plan
        self._trials = {
            'isi1': d['isi1'].tolist(),
            'stim': self.get_obj_url(),
            'stim_dur': d['stim_dur'].tolist(),
            'isi2': d['isi2'].tolist()
        }

    def sim_analysis(self):
        df = pandas.DataFrame(self.exp_plan)
        df.sort_values(by=['subjid', 'order', 'obj', 'imgno', 'stim_dur'], inplace=True)
        gr = df.groupby(['obj', 'objno', 'imgno']).groups.keys()
        fs = {}
        for obj, objno, imgno in gr:
            a = .1 #+ #.1 * np.random.random()
            b = .1 + objno/10. #+ #.1 * np.random.random()
            lam = .05 + imgno/100. #+ .1 * np.random.random()
            fs[(obj,imgno)] = (a,b,lam)

        def accf(row):
            a, b, lam = fs[(row.obj, row.imgno)]
            x = row.stim_dur / 1000.
            acc = .5 + (.5 - lam) / (1 + np.exp(-(x-a)/b))
            return acc

        df.acc = df.apply(accf, axis=1)
        df.acc = df.acc.astype(float)
        print(df[df.qe==False].groupby(['obj', 'imgno', 'stim_dur']).acc.mean())
        print(df[df.qe==True].groupby(['obj', 'imgno', 'stim_dur']).acc.mean())
        import pdb; pdb.set_trace()
        sel = df.obj.isin(df.obj.unique()[:2]) & \
              df.imgno.isin(df.imgno.unique()[:3])
        sns.factorplot(x='stim_dur', y='acc', col='obj', row='imgno',
                       data=df[sel], kind='point')
        sns.plt.show()

    def updateDBwithHITs(self, verbose=False, overwrite=False):
        hitids = self.get_hitids()
        coll = self.collection
        idx = 0
        with open('snodgrass_synonyms.txt') as f:
            wordlist = [w.strip('\n\r') for w in f.readlines()]

        for subjid, hitid in enumerate(hitids):
            subj_data = self.getHITdata(hitid, full=False)

            coll.ensure_index([
                ('WorkerID', pymongo.ASCENDING),
                ('Timestamp', pymongo.ASCENDING)],
                unique=True)

            assert len(subj_data) == 1

            for subj in subj_data:
                assert isinstance(subj, dict)

                data = zip(subj['ImgOrder'], subj['Response'], subj['RT'])
                for k, (img, resp, rt) in enumerate(data):
                    stim = self.exp_plan[idx]
                    corr_resps = [stim['name']]
                    try:
                        corr_resps += stim['synonyms'].split(', ')
                    except:
                        pass
                    try:
                        corr_resps += stim['synonyms_extra'].split(', ')
                    except:
                        pass
                    # fix for incomplete answers
                    if resp not in wordlist:
                        poss_resps = [w for w in wordlist if w.startswith(resp)]
                        if len(poss_resps) == 1:
                            resp = poss_resps[0]
                        else:
                            print('Multiple possible answers found for {}:'.format(stim['name']))
                            for i, res in enumerate(poss_resps):
                                print(i+1, res, sep=' - ')
                            choice = raw_input('Choose which one to use (default is 1): ')
                            if choice == '': choice = '1'
                            resp = wordlist[int(choice)]

                    self.exp_plan['subj_resp'][idx] = resp
                    self.exp_plan['acc'][idx] = resp in corr_resps
                    self.exp_plan['rt'][idx] = rt
                    idx += 1

                try:
                    doc_id = coll.insert(subj, safe=True)
                except pymongo.errors.DuplicateKeyError:
                    if not overwrite:
                        warnings.warn('Entry already exists, moving to next...')
                        continue
                    if 'WorkerID' not in subj or 'Timestamp' not in subj:
                        warn("No WorkerID or Timestamp in the subject's "
                                "record: invalid HIT data?")
                        continue
                    spec = {'WorkerID': subj['WorkerID'],
                            'Timestamp': subj['Timestamp']}
                    doc = coll.find_one(spec)
                    assert doc is not None
                    doc_id = doc['_id']
                    if '_id' in subj:
                        _id = subj.pop('_id')
                        if verbose and str(_id) not in str(doc_id) \
                                and str(doc_id) not in str(_id):
                            print('Dangling _id:', _id)
                    coll.update({'_id': doc_id}, {
                        '$set': subj
                        }, w=0)

                if verbose:
                    print('Added:', doc_id)

                # if meta is None:
                #     continue

                # handle ImgData
                m = self.exp_plan[self.exp_plan['subjid'] == subjid]
                m['subjid'] += 30
                m = pandas.DataFrame(m).to_dict('records')
                coll.update({'_id': doc_id}, {'$set': {'ImgData': m}}, w=0)

def get_syns():
    set_of_words = []
    df = pandas.read_csv('../data/snodgrass_syns.csv', sep='\t')
    set_of_words = set(df.name.values.tolist()+
                       df.synonym[df.synonym.notnull()].values.tolist())
    set_of_words = sorted(set(set_of_words), key=str.lower)
    with open('snodgrass/web/snodgrass_synonyms.txt', 'wb') as w:
        for s in set_of_words:
            w.write(s + '\n')

def publish_images(kind):
    ims = glob.glob(os.path.join('..', 'img', kind) + '/*.png')
    access, secret = mturkutils.base.parse_credentials_file(section_name='MTurkCredentials')
    conn = boto.connect_s3(access, secret)
    bucket = conn.create_bucket('snodgrass-naming')
    for imno, fname in enumerate(ims):
        print('\r{}'.format(imno), end='')
        sys.stdout.flush()
        # if fname not in dataset.meta['filename']:
        #     bucket.delete_key(os.path.join('ims', os.path.basename(fname)))
        #     count += 1

        s3file = bucket.new_key(os.path.join('images', kind, os.path.basename(fname)))

        # s3file = bucket.new_key(os.path.join(kind, os.path.basename(fname)))
        # fname = os.path.join('masks', name) + '.png'
        s3file.set_contents_from_filename(fname, policy='public-read')


parser = argparse.ArgumentParser()
parser.add_argument('func')
parser.add_argument('-p', '--production', action='store_true')
parser.add_argument('--single', action='store_true')
parser.add_argument('--short', action='store_true')
parser.add_argument('-n', '--dry', action='store_true')
args, extras = parser.parse_known_args()
kwargs = {}
for kwarg in extras:
    k, v = kwarg.split('=')
    kwargs[k.strip('-')] = v

exp = SnodgrassNaming(sandbox=not args.production, single=args.single,
                      short=args.short, save=not args.dry)
print()

if args.func == 'create':
    exp.create_exp_plan()
elif args.func == 'prep':
    exp.createTrials()
    # shutil.rmtree(exp.tmpdir)
    exp.prepHTMLs()
elif args.func == 'upload':
    exp.createTrials()
    # shutil.rmtree(exp.tmpdir)
    exp.prepHTMLs()
    if not args.single and not args.short:
        exp.testHTMLs()
    exp.uploadHTMLs()
elif args.func == 'create_hits':
    exp.createTrials()
    # shutil.rmtree(exp.tmpdir)
    exp.prepHTMLs()
    if not args.single and not args.short:
        exp.testHTMLs()
    exp.uploadHTMLs()
    exp.createHIT(secure=True)

elif args.func == 'download':
    exp.updateDBwithHITs(**kwargs)
elif args.func == 'test_data':
    hitids = exp.get_hitids()
    print(hitids)
    data = exp.getHITdata(hitids[0], full=False)
    import pdb; pdb.set_trace()
else:
    try:
        getattr(exp, args.func)(**kwargs)
    except:
        eval(args.func)(**kwargs)
