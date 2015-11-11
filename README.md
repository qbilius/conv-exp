# Introduction

**Deep Neural Networks as a Computational Model for Human Shape Sensitivity**

*Jonas Kubilius, Stefania Bracci, and Hans P. Op de Beeck*

Theories of object recognition agree that shape is of primordial importance, but there is no consensus about how shape might be represented and so far attempts to implement a model of shape perception that would work with realistic stimuli have largely failed. Recent studies suggest that state-of-the-art convolutional ‘deep’ neural networks (DNNs) capture important aspects of human object perception. We hypothesized that these successes might be partially related to a human-like representation of object shape. Here we demonstrate that sensitivity for shape features, characteristic to human and primate vision, emerges in DNNs when trained for generic object recognition from natural photographs. We show that these models explain human shape judgments for several benchmark behavioral and neural stimulus sets on which earlier models failed. In particular, although never explicitly trained for, these models develop sensitivity to non-accidental properties that have long been implicated to form the basis for object recognition. Even more strikingly, when tested with a challenging stimulus set in which shape and category membership are dissociated, the most complex model architectures capture human shape sensitivity as well as some aspects of the category structure that emerges from human judgments. As a whole, these results indicate that convolutional neural networks not only learn physically correct representations of object categories but also develop perceptually accurate representational spaces of shapes. An even fuller model of human object representations might be in sight by training deep architectures for multiple tasks, which is so characteristic in human development.

# Quick Start

## Reproduce results reported in the paper

`python run.py report --bootstrap`

Note that this will take a **very long time** (half a day or so). It will be substantially faster without bootstrapping:

`python run.py report`

## Run various tasks

`python run.py <dataset> <run / compare> <task> <options>`

Available datasets:
- snodgrass
- hop2008
- fonts
- geons
- stefania

For available options, see `run.py`. For tasks, see functions in `run.py` of an individual dataset. For example,

`python run.py hop2008 compare corr_all --n`

# Dependencies

- psychopy_ext 0.6 (`pip install psychopy_ext`; not yet available)
- Caffe (http://caffe.berkeleyvision.org/)

# License

Copyright 2015 Jonas Kubilius ([klab.lt](http://klab.lt)),

Brain and Cognition, KU Leuven (Belgium)

[GNU General Public License v3 or later](http://www.gnu.org/licenses/)
