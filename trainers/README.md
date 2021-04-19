# Trainers

Trainers are used are package relevant modules, datamodules/datagenerators, and 
models into a unified training pipeline. As an input, trainer scripts take a
configuration file (found in `configs/`), which guides tells the trainer as to the model
being trained, the module supporting that training, and the data that it should be
trained on. All outputs and logging of a trainer script should be nested into folder structure
`logs/<datamodule>/<model>/<version>/`. The datamodule and model must be declared
explicitly in the configuration file being used, otherwise defaults will be automatically
used. The most important output of a training script is the model checkpoint.
Once a checkpoint has been validated, it may be archived manually and used later in an
experiment for evaluation or visualization purposes. Other outputs of a training session
include a copy of the configuration file, a summary of the model architecture, images
saved during training and validation, and all losses. This repo favours the use of 
Tensorboard for logging. Tensorboard logs can be viewed by running the following command
in a separate terminal:

```bash
user@cpu:~/novelty-detection$ tensorboard --logdir logs/[<datamodule>]/[<model>] 
```

Here, the `<datamodule>` and/or `<model>` specifiers are optional; if used Tensorboard will
only show the training outputs for those specific datamodules/models.
