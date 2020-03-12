import os

import argparse

from .datasets import PersonalityCaptions, DatasetLoader
from .misc import init_logging
from .train import generator_mle_train

args = argparse.Namespace()

args.run_id = "run_1"
args.base_dir = os.path.dirname(os.path.dirname(__file__))
args.data_dir = os.path.join(args.base_dir, "data", "personality_captions")
args.results_dir = os.path.join(args.base_dir, "results")
args.checkpoints_dir = os.path.join(args.results_dir, args.run_id, "checkpoints")
args.log_dir = os.path.join(args.results_dir, args.run_id, "logs")
args.overwrite_checkpoint_dir = False

args.do_train = True
args.do_eval = True
args.seed = 42

args.generator_embedding_units = 512
args.generator_attention_units = 512
args.generator_lstm_units = 512
args.generator_lstm_dropout = 0.2
args.generator_mle_learning_rate = 1e-4
args.generator_mle_grad_clipvalue = 5.
args.generator_mle_dsa_lambda = 1.
args.generator_mle_batch_size = 64
args.generator_mle_epochs = 20
args.generator_mle_logging_steps = 1
args.generator_mle_validate_steps = 1000

init_logging(args.log_dir)

if args.do_train:
    personality_captions = PersonalityCaptions(args.data_dir)
    dataset_loader = DatasetLoader(personality_captions)
    generator_mle_train(args, dataset_loader)

if args.do_eval:
    pass
