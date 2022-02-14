### Data preparation

#### Format description

Where:
- $PATH_AUDIO_FILES is the directory containing the audio files. The files should be arranged as below:
```
PATH_AUDIO_FILES  
│
└───speaker1
│   └───...
│         │   seq_11.{$EXTENSION}
│         │   seq_12.{$EXTENSION}
│         │   ...
│   
└───speaker2
    └───...
          │   seq_21.{$EXTENSION}
          │   seq_22.{$EXTENSION}
```

Please note that each speaker directory can contain an arbitrary number of subdirectories: the speaker label will always be retrieved from the top one.

- $PATH_CHECKPOINT_DIR in the directory where the checkpoints will be saved
- $TRAINING_SET is a path to a .txt file containing the list of the training sequences (see [here](https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb) for example)
- $VALIDATION_SET is a path to a .txt file containing the list of the validation sequences
- $EXTENSION is the extension of each audio file


#### Long-form recordings

WIP

#### Simulated long-form recordings

WIP

#### Audiobooks

WIP