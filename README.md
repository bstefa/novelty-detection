# Novelty Detection

This repository contains the research-level code used to complete my MASc: Image-based Novelty Detection for Lunar Exploration.

> For more details about the nature and outcomes of the research conducted herein feel free to checkout my thesis [here](https://spectrum.library.concordia.ca/id/eprint/988786/).

## Usage and Terminology

In this repository, the **experiments** form the primary research scripts; this is a good place to start if you're only interested in the results or if you prefer to read code over plain-text explanations. Experiments vary in scope from single model performance benchmarking to multi-model testing, comparison, and visualization. Typically, an experiment will use a single (or set of) **module**(s) and **dataset**(s) (sometimes referred to as datamodules or datagenerators depending on the task).

This repository is structured such that a module contains the code needed to carry out basic training functionality, and a dataset defines how the data is imported, processed, curated, and batched. A module's member functions include definitions of the training, validation, and test steps, and preperation of objective functions and optimizers. Most importantly, each module is parameterized by a **model**, which may be a CAE, a VAE, an AAE, or PCA-based. This seperataion of concerns enables a number of benefits: it allows models to be swapped in and out of their containing module with ease, allowing one to train, evaluate, and visualize many different combinations of algorithms and datasets; it allows models to be serialized independently, easing the process of preparing them for production; lastly, it helps keep the codebase modular and clean (though confusing for those new to the repo).

Training is conducted with **trainers**, scripts that synthesize the various components mentioned above and set up peripheral functionality, such as logging, callbacks, and importing configurations (**configs**).
