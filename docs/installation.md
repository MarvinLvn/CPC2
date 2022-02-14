### Installation

The code relies on [pyannote-audio](https://github.com/pyannote/pyannote-audio), a python package 
for speaker diarization: speech activity detection, speaker change detection, speaker embedding.

```bash
# Step 1: git clone the voice type classifier repo as well as pyannote-audio dependency
$ git clone https://github.com/MarvinLvn/CPC2.git
$ cd CPC2

# Step 2 : create conda env called "cpc2", installing all the required dependencies
$ conda env create -f env.yml
```

Make sure [sox](http://sox.sourceforge.net/) is installed too.
Once everything has been installed, you can start [preparing the data](../docs/data_preparation.md) or [training the model](../docs/training_and_eval.md).

