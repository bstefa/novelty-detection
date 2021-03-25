# Novelty Detector

## Repository Structure and Terminology

In this repository, the **experiments** form the primary research scripts; this is a good place to start if you're only interested in the results or if you prefer to read code pver plain-text- explanations. Experiments vary in scope from single model performance benchmarking to multi-model testing, comparison, and visualization. Typically, an experiment will use a single (or set of) **module**(s) and **dataset**(s) (sometimes referred to as datamodules or datagenerators depending on the task).

This repository is structured such that a *module* contains the code needed to carry out basic training functionality, and a *dataset* defines how the data is imported, processed, curated, and batched. A *module*'s member functions include definitions of the training, validation, and test steps, and preperation of objective functions and optimizers. Most importantly, each *module* is parameterized by a **model**, which may be a CAE, a VAE, PCA-based, or a number of others, you can read more about the support models [here](https://github.com/brahste/novelty-detection/tree/main/models). This seperataion of concerns enables a number of benefits: it allows *models* and *configurations* to be swapped in and out of their containing *module* with ease, allowing us to train, evaluate, and visualize many different combinations of algorithms and datasets; it allows *models* to be serialized independently, easing the process of preparing them for production; lastly, it helps keep the codebase modular and clean (though confusing for those new to the repo). Many *models* have been built to address the problem of novelty detection, some of them well-established, some of them very much research-level inquiries. 

Training is conducted with **trainers**, scripts that synthesize the various components mentioned above and set up peripheral functionality, such as logging, callbacks, and importing configurations (**configs**). A trainer is setup with a default config but alternative configs can be passed at the command line when calling the training script, for instance:

```
$ python trainers/train_cae_baseline.py config/cae/cae_baseline_mnist.yaml
```
