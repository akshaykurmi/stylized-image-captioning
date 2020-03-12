import logging
import os

import tensorflow as tf
from tqdm import tqdm

from .misc import set_seed
from .models import Generator, Encoder

logger = logging.getLogger(__name__)


@tf.function
def generator_mle_train_batch(args, encoder, generator, loss_fn, batch):
    images, captions, _ = batch
    with tf.GradientTape() as tape:
        encoder_output = encoder(images)
        predictions, attention_alphas = generator.train_mle_forward(encoder_output, captions)
        loss = loss_fn(captions, predictions)
        loss += args.generator_mle_dsa_lambda * tf.reduce_mean(1. - tf.reduce_sum(attention_alphas, axis=1) ** 2)
        gradients = tape.gradient(loss, encoder.trainable_variables + generator.trainable_variables)
    return loss, gradients


@tf.function
def generator_mle_val_loss(args, encoder, generator, loss_fn, batch):
    images, captions, _ = batch
    encoder_output = encoder(images)
    predictions, attention_alphas = generator.train_mle_forward(encoder_output, captions)
    loss = loss_fn(captions, predictions)
    loss += args.generator_mle_dsa_lambda * tf.reduce_mean(1. - tf.reduce_sum(attention_alphas, axis=1) ** 2)
    return loss


def generator_mle_train(args, dataset_loader):
    if os.path.exists(args.checkpoints_dir) and os.listdir(args.checkpoints_dir) and not args.overwrite_checkpoint_dir:
        raise ValueError("Checkpoints directory {} already exists and is not empty".format(args.checkpoints_dir))

    logger.info("***** Training Generator using MLE - Starting *****")

    set_seed(args.seed)

    logger.info("-- Loading training set")
    train_dataset = dataset_loader.load("train", batch_size=args.generator_mle_batch_size)
    val_dataset = dataset_loader.load("val", batch_size=args.generator_mle_batch_size)
    num_train_batches = tf.data.experimental.cardinality(train_dataset).numpy()

    encoder = Encoder()
    generator = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.generator_mle_learning_rate,
                                         clipvalue=args.generator_mle_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = 0
    for epoch in args.generator_mle_epochs:
        logger.info(f"Epoch: {epoch + 1}/{args.generator_mle_epochs}")
        for step, train_batch in enumerate(tqdm(train_dataset, total=num_train_batches, desc="Batch", unit="batch")):
            loss, gradients = generator_mle_train_batch(args, encoder, generator, loss_fn, train_batch)
            global_step += 1
            optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + generator.trainable_variables))
            if global_step % args.generator_mle_logging_steps == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("generator_mle_loss", loss, step=global_step)
            if (global_step == args.generator_mle_logging_steps or
                    global_step % args.generator_mle_validate_steps == 0):
                losses = [generator_mle_val_loss(args, encoder, generator, loss_fn, val_batch)
                          for val_batch in val_dataset]
                with val_summary_writer.as_default():
                    tf.summary.scalar("generator_mle_loss", tf.reduce_mean(losses), step=global_step)

    logger.info("***** Training Generator using MLE - Ended *****")
