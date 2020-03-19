# Stylized Image Captioning using Conditional Sequence GANs

Course project for CS7140 - Advanced Machine Learning - Spring 2020

#### Environment Setup
Python 3.6 is required to execute the code. After creating a Python 3.6 virtual environment using Virtualenv or Miniconda, run the following to install the dependencies:
```bash
pip install -r requirements.txt
```

#### Training the model
There are several stages involved in training the model.
These are controlled by the boolean flags - ``run_download_dataset, run_generator_pretraining, run_discriminator_pretraining, run_adversarial_training`` in the file ``src/main.py``.
Enable the stages you wish to execute and run the following:
```bash
python -m src.main
```
This will begin the training process.
Checkpoints and tensorboard logs will automatically be created in the ``results`` directory in the root of the project.
To monitor training progress, execute:
```bash
tensorboard --logdir results/
```