# Early phonetic learning from ecological audio: domain-general versus domain-specific mechanisms

WORK IN PROGRESS

This is the git repository associated to our PNAS publication: *Early phonetic learning from ecological audio: domain-general versus domain-specific mechanisms*
In this repository, you'll find all the necessary code for training a contrastive predictive coding (CPC) model from raw speech.
Adapted and modified from the publication [Unsupervised Pretraining Transfers well Across Languages](https://arxiv.org/abs/2002.02848), whose associated git repository can be found [here](https://github.com/facebookresearch/CPC_audio).

### How to use ?

1) [Installation](./docs/installation.md)
2) [Data preparation](./docs/data_preparation.md)
3) [Training and Evaluation](./docs/training_and_eval.md)

### References

Main paper:

```
@misc{rivire2020unsupervised,
    title={Early phonetic learning from ecological: domain-general versus domain-specific },
    author={Marvin Lavechin and Maureen de Seyssel and Marianne Métais and Florian Metze and Abdelrahman Mohamed and Hervé Bredin and Emmanuel Dupoux and Alejandrina Cristia},
    year={2022},
    eprint={WIP},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

Code from which our work has been based on:

```
@misc{rivire2020unsupervised,
    title={Unsupervised pretraining transfers well across languages},
    author={Morgane Rivière and Armand Joulin and Pierre-Emmanuel Mazaré and Emmanuel Dupoux},
    year={2020},
    eprint={2002.02848},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```