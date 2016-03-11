# Introduction

**Deep Neural Networks as a Computational Model for Human Shape Sensitivity**

*Jonas Kubilius, Stefania Bracci, and Hans P. Op de Beeck*

Theories of object recognition agree that shape is of primordial importance, but there is no consensus about how shape might be represented and so far attempts to implement a model of shape perception that would work with realistic stimuli have largely failed. Recent studies suggest that state-of-the-art convolutional ‘deep’ neural networks (DNNs) capture important aspects of human object perception. We hypothesized that these successes might be partially related to a human-like representation of object shape. Here we demonstrate that sensitivity for shape features, characteristic to human and primate vision, emerges in DNNs when trained for generic object recognition from natural photographs. We show that these models explain human shape judgments for several benchmark behavioral and neural stimulus sets on which earlier models failed. In particular, although never explicitly trained for, these models develop sensitivity to non-accidental properties that have long been implicated to form the basis for object recognition. Even more strikingly, when tested with a challenging stimulus set in which shape and category membership are dissociated, the most complex model architectures capture human shape sensitivity as well as some aspects of the category structure that emerges from human judgments. As a whole, these results indicate that convolutional neural networks not only learn physically correct representations of object categories but also develop perceptually accurate representational spaces of shapes. An even fuller model of human object representations might be in sight by training deep architectures for multiple tasks, which is so characteristic in human development.


# Quick Start

## Reproduce results reported in the paper

*(Note that you need to [set up all dependencies first](#setup))*

`python run.py report --bootstrap`

This will take a **very long time** (half a day or so). It will be substantially faster (but still at least several hours) without bootstrapping:

`python run.py report`

## Get datasets

`python run.py download_datasets`

Note that if you run `report`, datasets will be downloaded automatically.

## Run various tasks

`python run.py <dataset> <run / compare> <task> <options>`

Available datasets:
- snodgrass (Exp. 1)
- hop2008 (Exp. 2a)
- fonts (Exp. 2b)
- geons (Exp. 3)
- stefania (Exp. 4)

For available options, see `run.py`. For tasks, see functions in `run.py` of an individual dataset. For example,

`python run.py hop2008 compare corr_all --n`


# Setup

## General

All experiments were done on Ubuntu (14.04 LTS and later 12.04 LTS). In principle, everything could be done on Windows or Mac OS X, except that setting up Caffe might be a challenge. But you may want to run models separately anyways, especially if your local machine does not have a GPU. See [Deep models](#deep-models) for more information.

Also, I use ImageMagick during runtime for converting and processing images, which may also cause challenges to install if you don't have it yet (Ubuntu comes packaged with it). If ImageMagick is not an option, all stimuli are available on [OSF](https://osf.io/jf42a/), though that means you will have to ask us for them in person and also download them manually.

## Python

All experiments were done using Python 2.7, but the code should compatible with Python 3+ as I included `__future__` imports (but this has not been tested). I recommend using [conda](http://conda.pydata.org/miniconda.html) because it makes it easy to install all dependencies.

Specific packages:

- `psychopy_ext` 0.6 (`pip install psychopy_ext`; not yet available), which requires numpy, scipy, pandas, seaborn
- `ntlk` (make sure to get `nltk` data: `import nltk; nltk.download()`
    )

## Stimuli

Images are supposed to be stored in the `img` folder in each experiment. Typically, there are several versions of stimuli, each in their corresponding subfolder (e.g., `img/alpha`).

Stimuli for Exp. 2a (from Op de Beeck et al., 2008) are included in the repository.

Stimuli for Exps. 1, 2b, and 3 can be easily obtained by running

`python run.py download_datasets`

Note that that stimuli for Exp. 1 (Snodgrass and Vanderwart, 1980), adapted by Rossion & Pourtois (2004), are available in the obsolete PCT format from [Tarr Lab](http://wiki.cnbc.cmu.edu/Objects). So instead an PNG version is downloaded here.

Stimuli for Exp. 4 (from Bracci & Op de Beeck, 2016) are subject to copyright and cannot be shared so easily. Please email one of the people listed in the publication to obtain the URL to this stimulus set. If you don't want to bother, just remove `'stefania'` in the `ALL_EXPS` variable on line 30 in `base.py`.

## Models

Some models are easier to obtain than others so if some of them are giving you a headache, consider removing them from lines 27-29 in `base.py`.

### Python models

Pixelwise, GaborJet, HoG, and HMAX'99 are included in `psychopy_ext`.

### MATLAB models

Several models have not been ported to Python, so you need MATLAB to run them. I interface with MATLAB from Python using `matlab_wrapper` library, which is a `psychopy_ext` dependency. In my experience, it works only with some MATLAB versions. If `matlab_wrapper` is not working for you, you may also want to try a newer MATLAB version that also have [their own API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

To obtain PHOG and PHOW, run

`python run.py download_models`

After you download these models, make sure to add them to your path. One easy and permanent way to do that is to include the following line in your `.bashrc` file:

`export PHOG=<path to model>`

You will have to download HMAX-PNAS and HMAX-HMIN manually because their authors ask you to provide an email where a unique download link is provided.

**HMAX-HMIN**

- [Download from here](http://cbcl.mit.edu/jmutch/hmin/)
- Compile: `matlab -r "mex example.cpp"`
- Add HMAX_HMIN to path as explained above

**HMAX-PNAS**

- Download [FHLib](http://www.mit.edu/~jmutch/fhlib/)
- Download [HMAX-PNAS](http://cbcl.mit.edu/software-datasets/pnas07/index.html), which you need to unpack under FHLib.
- Compile:
- Add HMAX_PNAS to path as explained above

### Caffe (deep) models

In principle, to run a deep network, you will need a GPU. It is not strictly necessary (you can run computations on a CPU too: `python run.py report --mode cpu`), but it would take you forever. For a GPU, at the very least you need a simple NVIDIA CUDA-enabled GPU with like 1GB RAM. It will run CaffeNet, but that's about it. For VGG-19 and GoogLeNet, I think 6GB RAM is the minimum.


I used [Caffe](http://caffe.berkeleyvision.org/) for running deep models. Caffe [installation is doable on a Ubuntu 14.04 machine](http://caffe.berkeleyvision.org/install_apt.html) with administrator permissions and if you have NVIDIA drivers set up. If the drivers are not set up (that is, you have the default Ubuntu drivers), then you will have to go through a pain installing them and the risk of losing your graphical environment.

I have also installed Caffe on a cluster running CentOS, and it was hard.

To obtain the models (you need to be in `$CAFFE` directory):

- CaffeNet: `python scripts/download_model_binary.py models/bvlc_reference_caffenet `
- VGG-19:

```bash
./scripts/download_model_from_gist.sh 3785162f95cd2d5fee77
mv models/3785162f95cd2d5fee77 models/vgg-19
cd models/vgg-19
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
```

- GoogLeNet: `python scripts/download_model_binary.py models/bvlc_googlenet`


# License

Copyright 2015-2016 Jonas Kubilius ([klab.lt](http://klab.lt))

Brain and Cognition, KU Leuven (Belgium)

McGovern Institute for Brain Research, MIT (USA)

[GNU General Public License v3 or later](http://www.gnu.org/licenses/)
