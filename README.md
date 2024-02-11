# Visual Novelty Detection Framework

![image-20240211115342993](figures/coverphoto1.png)

*Abstract*

This repository was developed as part of a research endeavour in the field of image-based novelty detection, with a focus on applications in lunar exploration. It provides a modular and flexible framework for image processing and machine learning research, accommodating a range of  experiments from individual model evaluations to comprehensive multi-model analyses. The framework is designed for ease of use and modularity, featuring parameterizable modules and ready to use models, including CAE, VAE, and PCA. This design facilitates straightforward model swapping and efficient serialization, ensuring a clean and understandable codebase. It serves as a crucial resource for researchers and practitioners in the domains of image processing and space exploration technology.

> [!tip]
>
> When citing this work, you can use the **Cite this repository** button to the right.

*Related publications*

1. Stefanuk, B. and Skonieczny, K., (2022). Novelty detection in rover‑based planetary surface images using autoencoders. Front. Robot. AI, 9:974397. [doi.org/10.3389/frobt.2022.974397](https://doi.org/10.3389/frobt.2022.974397)

2. Stefanuk, B. (2021). Image-based novelty detection for lunar exploration. Master’s thesis. Montreal, Canada. Concordia University.



## Usage and Terminology

In this repository, the **experiments** form the primary research scripts; this is a good place to start if you're only interested in the results or if you prefer to read code over plain-text explanations. Experiments vary in scope from single model performance benchmarking to multi-model testing, comparison, and visualization. Typically, an experiment will use a single (or set of) **module**(s) and **dataset**(s) (sometimes referred to as datamodules or datagenerators depending on the task).

This repository is structured such that a module contains the code needed to carry out basic training functionality, and a dataset defines how the data is imported, processed, curated, and batched. A module's member functions include definitions of the training, validation, and test steps, and preperation of objective functions and optimizers. Most importantly, each module is parameterized by a **model**, which may be a CAE, a VAE, an AAE, or PCA-based. This seperataion of concerns enables a number of benefits: it allows models to be swapped in and out of their containing module with ease, allowing one to train, evaluate, and visualize many different combinations of algorithms and datasets; it allows models to be serialized independently, easing the process of preparing them for production; lastly, it helps keep the codebase modular and clean (though confusing for those new to the repo).

Training is conducted with **trainers**, scripts that synthesize the various components mentioned above and set up peripheral functionality, such as logging, callbacks, and importing configurations (**configs**).
