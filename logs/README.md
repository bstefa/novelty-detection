# Logs

Any outputs generated during training are, by default, placed in this directory. During a training session outputs are saved with the following (example) directory structure:
```bash
logs
├── CuriosityDataModule
│   ├── BaselineAAE
│   │   ├── version_1
│   │   │   ├── checkpoints
│   │   │   │   ├── last.ckpt
│   │   │   │   └── val_r_loss=0.45-epoch=34.ckpt
│   │   │   ├── configuration.yaml
│   │   │   ├── events.out.tfevents.1620396411.Rufus.18731.0
│   │   │   ├── hparams.yaml
│   │   │   ├── images
│   │   │   │   ├── epoch=0-step=3.png
│   │   │   │   ├── ...
│   │   │   │   └── epoch=35-step=1301.png
│   │   │   └── model_summary.txt   
│   │   ... 
│   ...
...
```

For this training log, the configuration passed to the trainer is logged in the *configuration.yaml* file (*hparam.yaml* is an empty artifact). A textual description of the model's architecture is generated in *model_summary.txt*, this file is especially useful for those wishing the replicate the findings or implement a variation of the model. Details such as batch normaization parameters, dropout rates, and convolutional kernel size are also documented in the model summary. 

While validating a training session it is typical to train past the lowest target loss, this early stopping ensures that the model doesn't overfit the data and allows to training to adhere to performance requirement established early in the design cycle. To this end, the last training epoch before which the stopping patience was triggered is logged to the *checkpoints/last.ckpt* file, while the lowest achieved target loss is logged to the *checkpoints/<loss>=<value>-epoch=<integer>.ckpt (e.g. *val_r_loss=0.45-epoch=34.ckpt* in the directory tree above).
