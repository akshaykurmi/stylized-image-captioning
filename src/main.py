import os
import shutil

import argparse

from .datasets import PersonalityCaptions, DatasetLoader
from .train import pretrain_generator, pretrain_discriminator, adversarially_train_generator_and_discriminator
from .utils import init_logging

args = argparse.Namespace()

args.run_id = "run_1"
args.base_dir = os.path.dirname(os.path.dirname(__file__))
args.data_dir = os.path.join(args.base_dir, "data", "personality_captions")
args.results_dir = os.path.join(args.base_dir, "results")
args.run_dir = os.path.join(args.results_dir, args.run_id)
args.checkpoints_dir = os.path.join(args.run_dir, "checkpoints")
args.log_dir = os.path.join(args.run_dir, "logs")
args.overwrite_run_results = True

args.run_download_dataset = False
args.run_generator_pretraining = True
args.run_discriminator_pretraining = False
args.run_adversarial_training = False
args.run_evaluation = False

args.seed = 42
args.max_seq_len = 20

args.generator_embedding_units = 512
args.generator_attention_units = 512
args.generator_lstm_units = 512
args.generator_lstm_dropout = 0.2
args.discriminator_embedding_units = 512
args.discriminator_lstm_units = 512

args.generator_pretrain_scheduled_sampling_initial_rate = 1
args.generator_pretrain_scheduled_sampling_k = 3500
args.generator_pretrain_learning_rate = 1e-4
args.generator_pretrain_grad_clipvalue = 5.
args.generator_pretrain_dsa_lambda = 0.9
args.generator_pretrain_batch_size = 64
args.generator_pretrain_epochs = 20
args.generator_pretrain_logging_steps = 1
args.generator_pretrain_validate_steps = 1000
args.generator_pretrain_checkpoint_steps = 50

args.discriminator_pretrain_learning_rate = 1e-4
args.discriminator_pretrain_grad_clipvalue = 5.
args.discriminator_pretrain_batch_size = 64
args.discriminator_pretrain_neg_sample_weight = 0.5
args.discriminator_pretrain_epochs = 10
args.discriminator_pretrain_logging_steps = 1
args.discriminator_pretrain_validate_steps = 1000
args.discriminator_pretrain_checkpoint_steps = 50

args.generator_adversarial_learning_rate = 1e-4
args.generator_adversarial_grad_clipvalue = 5.
args.generator_adversarial_logging_steps = 1
args.generator_adversarial_batch_size = 64
args.generator_adversarial_dsa_lambda = 0.9
args.discriminator_adversarial_learning_rate = 1e-4
args.discriminator_adversarial_grad_clipvalue = 5.
args.discriminator_adversarial_logging_steps = 1
args.discriminator_adversarial_batch_size = 22
args.discriminator_adversarial_neg_sample_weight = 0.5
args.adversarial_rounds = 10000
args.adversarial_validate_rounds = 50
args.adversarial_checkpoint_rounds = 5
args.adversarial_g_steps = 1
args.adversarial_d_steps = 3
args.adversarial_rollout_n = 10
args.adversarial_rollout_update_rate = 1

init_logging(args.log_dir)

personality_captions = PersonalityCaptions(args.data_dir)
dataset_loader = DatasetLoader(personality_captions, args.max_seq_len)

if args.overwrite_run_results:
    shutil.rmtree(args.run_dir, ignore_errors=True)

if args.run_download_dataset:
    personality_captions.download()

if args.run_generator_pretraining:
    pretrain_generator(args, dataset_loader)

if args.run_discriminator_pretraining:
    pretrain_discriminator(args, dataset_loader)

if args.run_adversarial_training:
    adversarially_train_generator_and_discriminator(args, dataset_loader)

if args.run_evaluation:
    pass
