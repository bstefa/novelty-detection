Experiments
===========

Experiments perform the role of packaging the module, model, dataset, and any monitoring, serializing, or visualizations tools. Experiments form the cornerstone of this research and are used to handle/standardize differences between inputs, operations, and outputs of different techniques. For example, an algorithm that uses PCA for novelty detection will differ substantially from one that uses a variational autoencoder. Each experiment should serialize, or otherwise prepare, a function that can be used later for evaluation. Ideally the differences between inputs and outputs of resulting functions are similar enough that they can be evaluated in tandem, or with only minor alterations, and thus can be easily compared.

## Some pointers
1. Experiment names should be explicit, better to know what the experiment does directly from its name than save a few key strokes and keep it ambiguous.
