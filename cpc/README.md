# Repository's architecture

train.py : main script

dataset.py : defintion of the Librispeech dataset

model.py : Basic encoders and AR models

feature_loader.py: different tools to load and save a CPC model.

transformers.py: an implementation of transformers

unit_tests.py : unit tests

criterion/: definition of the training criterions. Three criterion are currently available: CPC (unsupervised), speaker classification and phone classification.

distributed_learning/: tools to run a training with submitit (FAIR only)

eval/: evaluation scripts.

utils/: system utilities and misc.
