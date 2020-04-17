# Stylized Image Captioning using Conditional Sequence GANs

Course project for CS7140 - Advanced Machine Learning - Spring 2020

### Team
- Akshay Kurmi
- Emmanuel Ojuba

### Environment Setup
Python 3.6 is required to execute the code. After creating a Python 3.6 virtual environment using Virtualenv or Miniconda, run the following to install the dependencies
```bash
pip install -r requirements.txt
```

### Downloading
To download the Personality Captions dataset and the images, run
```bash
python -m src.main --run_download_dataset
```
Note that this takes several hours to run as the images need to be downloaded one by one.

### Caching Image Features
To optimize our pipeline, we pre-compute and cache the feature maps from the Resnet101 model.
Running the following command will preprocess and cache the dataset.
```bash
python -m src.main --run_cache_dataset
```
To override a previously cached dataset, add the flag `--overwrite_cached_dataset` to the command. 

### Training the model
First, pretrain the generator
```bash
python -m src.main --run_generator_pretraining
```
Then, pretrain the discriminator
```bash
python -m src.main --run_discriminator_pretraining
```
Finally, adversarially train the generator and the discriminator
```bash
python -m src.main --run_adversarial_training
```
These commands control the training process.
Checkpoints and tensorboard logs will automatically be created in the ``results`` directory in the root of the project.
To monitor training progress, execute
```bash
tensorboard --logdir results/
```
Add the `--run_id` flag to every command to specify a run ID.
To overwrite a runs results while training, add the `--overwrite_run_results` flag.
To train the stylized versions of the models, add the `--stylize` flag.

### Evaluating the models
To evaluate stylized models, add the `--stylize` flag.
To compute the human baseline scores, run
```bash
python -m src.main --run_human_evaluation
```
Metrics can be computed for several checkpoints at a time.
To compute metrics for a generator model, run
```bash
python -m src.main --run_evaluation --checkpoints_to_evaluate <ckpt1>,<ckpt2>,<ckpt3>
```
