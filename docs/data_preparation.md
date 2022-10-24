### Data preparation

#### Audio samples

Audiobooks

https://user-images.githubusercontent.com/11290637/197494765-6c16785e-42c8-44c0-bbdc-717e8ba0818f.mp4


https://user-images.githubusercontent.com/11290637/197494794-32f93d9b-9d37-453c-9cc2-6026063481f0.mp4




Simulated long-forms


https://user-images.githubusercontent.com/11290637/197494814-59077581-6d3b-4605-993d-92b98fd2032b.mp4


https://user-images.githubusercontent.com/11290637/197494819-2adc296d-3c42-49c1-9a2e-31840b2097ac.mp4



Long-forms


https://user-images.githubusercontent.com/11290637/197494830-3f29d2e6-b3be-46de-97d8-ddabc4f66b6d.mp4



https://user-images.githubusercontent.com/11290637/197494841-e58860ab-c47e-4b88-9899-8b7cddaf4e42.mp4



#### Format description

The training set should be arranged as follows:

```
PATH_AUDIO_FILES  
│
└───class1
│   └───...
│         │   seq_11.wav
│         │   seq_12.wav
│         │   ...
│   
└───class2
    └───...
          │   seq_21.wav
          │   seq_22.wav
```

The `class_i` information can then be used to sample sequences when training the model.
For instance, if `class_i` corresponds to speakers, then batch sequences can be drawn within- or across-speakers.
Please note that each class directory can contain an arbitrary number of subdirectories: the class label will always be retrieved from the top one.

#### Audiobooks

WIP

#### Long-form recordings

WIP

#### Simulated long-form recordings

Once audiobooks have been downloaded, and their speech segments have been extracted, you can ...

### Utility scripts

