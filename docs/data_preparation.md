### Data preparation

#### Format description

The training set should be arranged as follows:

```
PATH_AUDIO_FILES  
│
└───class1
│   └───...
│         │   seq_11.{$EXTENSION}
│         │   seq_12.{$EXTENSION}
│         │   ...
│   
└───class2
    └───...
          │   seq_21.{$EXTENSION}
          │   seq_22.{$EXTENSION}
```

The `class_i` information can then be used to sample sequences when training the model.
For instance, if `class_i` corresponds to speakers, then batch sequences can be drawn within- or across-speakers.
Please note that each class directory can contain an arbitrary number of subdirectories: the class label will always be retrieved from the top one.

#### Audiobooks

WIP

#### Long-form recordings

WIP

#### Simulated long-form recordings

WIP

