# Trainers

Trainers are used are package relevant modules, datamodules/datagenerators, and 
models into a unified training pipeline. As an input, trainer scripts take a
configuration file (found in `configs/`), which tells the trainer about the model
being trained, the module supporting that training, and the data that it should be
trained on.

A trainer is setup with a default configuration but alternative configurations can be passed at the command line when calling the training script, for instance to run a trainer use some variation of the following command:

```bash
$ python trainers/train_cae_baseline.py config/cae/cae_baseline_mnist.yaml
```

> Restrictions have been placed in the code to fail before training if a trainer-configuration combo isn't compatible.

All outputs and logging resulting from a training session are nested into the folder structure `logs/<datamodule>/<model>/<version>/` as specified in the configuration file. The datamodule and model must be declared explicitly in the configuration file being used, otherwise defaults will be automatically used.

The most important output of a training script is the model checkpoint. Once a checkpoint has been validated, it may be archived manually and used later in an experiment for evaluation or visualization purposes. Other outputs of a training session include a copy of the configuration file, a summary of the model architecture, images saved during training and validation, and all losses. This repo favours the use of 
Tensorboard for logging. Tensorboard logs can be viewed by running the following command:

```bash
$ tensorboard --logdir logs/[<datamodule>]/[<model>] 
```

Here, the `<datamodule>` and/or `<model>` specifiers are optional; if specified, Tensorboard will only show the training outputs for those specific datamodules/models.

## Notes

- Models that are kept for long-term reference are renamed with the naming convention: *archived_vX_yyyy-mm-dd*. Where the *X* is the version number. In this way the path to the model and it's training artifacts is *logs/datamodule/archived_vX_yyyy-mm-dd*. This convention is used to access models in the *experiments*.
