# NLB Technical Walkthrough

This tutorial covers the process of participating in NLB'21, from dataset download to submission on EvalAI. 

The main notebook is `nlb_technical_walkthrough.ipynb`, which is designed to be run on Colab but can also be run locally. The file `train.py` contains the script we used to train the RNN used in the notebook, saved in `pretrained_rnn.ckpt`.

## Setup

To run on Colab, simply click the `Open in Colab` button at the top of the notebook and proceed.

To run locally, we recommend setting up a conda environment with Python 3.7:
```
conda create --name nlb python=3.7
```

Then, you should install the following dependencies:
```
pip install git+https://github.com/neurallatents/nlb_tools.git
pip install torch
pip install dandi
pip install evalai
```

## Other resources

We have a number of other tutorials and example scripts covering a variety of topics:
* The notebooks in the [`nlb_tools` repo](https://github.com/neurallatents/nlb_tools) demonstrate application of classical methods like spike smoothing, GPFA, and SLDS to NLB'21.
* Andrew Sedler's [nlb-lightning](https://github.com/arsedler9/nlb-lightning) package provides a convenient framework to develop and evaluate PyTorch Lightning models for NLB'21.
