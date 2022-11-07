### Installation

First, make sure [sox](http://sox.sourceforge.net/) is installed.

You can clone this repo, and install all required dependencies by running:

```bash
# Step 1: git clone this repo
git clone https://github.com/MarvinLvn/CPC2.git
cd CPC2

# Step 2 : create conda env called "cpc2", installing all the required dependencies
conda env create -f environment.yml

# Step 3: install WavAugment
git clone https://github.com/facebookresearch/WavAugment.git && cd WavAugment
git checkout 357b2f9f09832cbe64ff76633eea8dbd5f1e97d1
pip install -e .
```

Once everything has been installed, you can start [preparing the data](../docs/data_preparation.md) or [training the model](../docs/training_and_eval.md).

