Modules
=======

For the purposes of this research a **module** is an experimental component (class) that is parameterized by a dataset and a model---that is, it takes as its first parameter a dataset (e.g. a data-module or data-generator), as its second parameter a model (e.g. a CAE or PCA routine), and as its third parameter a configuration dictionary. It should be a python class that is self-contained with all methods needed to run an experiment given the dataset and model selected.

To the extent possible, a module should be agnostic to the dataset, model, and any combination of the thereof. In practice, this isn't entirely feasible as sometimes specific dependencies for an algorithm prohibit other the use of certain datasets or models. For instance, a module that is used to train, test, and log results PyTorch models (e.g. CAE, VCAE) is not suitable for models built on the Scikit-Learn framework as one requires the data to be loaded into GPU memeory and the other into CPU memory. Complications of this sort could potentially be abstracted away within the module itself, but it is often more practical and clear to create a new module altogether.

## List of compatible models and datasets
