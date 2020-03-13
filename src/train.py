import logging
import os

import tensorflow as tf
from tqdm import tqdm

from .models import Generator, Encoder, Discriminator
from .utils import set_seed, MultiCheckpointManager

logger = logging.getLogger(__name__)


@tf.function
def generator_mle_train_batch(encoder, generator, loss_fn, batch, dsa_lambda):
    images, captions = batch
    with tf.GradientTape() as tape:
        encoder_output = encoder(images)
        predictions, attention_alphas = generator.train_mle_forward(encoder_output, captions)
        loss = loss_fn(captions, predictions)
        # TODO: Include doubly stochastic attention loss?
        # loss += dsa_lambda * tf.reduce_mean(1. - tf.reduce_sum(attention_alphas, axis=1) ** 2)
        gradients = tape.gradient(loss, encoder.trainable_variables + generator.trainable_variables)
    return loss, gradients


@tf.function
def generator_mle_val_loss(encoder, generator, loss_fn, batch, dsa_lambda):
    images, captions = batch
    encoder_output = encoder(images)
    predictions, attention_alphas = generator.train_mle_forward(encoder_output, captions)
    loss = loss_fn(captions, predictions)
    # TODO: Include doubly stochastic attention loss?
    # loss += dsa_lambda * tf.reduce_mean(1. - tf.reduce_sum(attention_alphas, axis=1) ** 2)
    return loss


@tf.function
def discriminator_mle_train_batch(encoder, discriminator, loss_fn, batch):
    images, captions, labels, sample_weight = batch
    with tf.GradientTape() as tape:
        encoder_output = encoder(images)
        predictions = discriminator(encoder_output, captions)
        loss = loss_fn(labels, predictions, sample_weight=sample_weight)
        gradients = tape.gradient(loss, discriminator.trainable_variables)
    return loss, gradients


@tf.function
def discriminator_mle_val_loss(encoder, discriminator, loss_fn, batch):
    images, captions, labels, sample_weight = batch
    encoder_output = encoder(images)
    predictions = discriminator(encoder_output, captions)
    loss = loss_fn(labels, predictions, sample_weight=sample_weight)
    return loss


def generator_mle_train(args, dataset_loader):
    logger.info("***** Training Generator using MLE - Starting *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    encoder = Encoder()
    generator = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.generator_mle_learning_rate,
                                         clipvalue=args.generator_mle_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, args.overwrite_checkpoint_dir, {
        "encoder": {"encoder": encoder},
        "generator": {"generator": generator},
        "generator_mle_params": {"optimizer": optimizer, "global_step": global_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    train_dataset = dataset_loader.load_generator_dataset("train", batch_size=args.generator_mle_batch_size)
    val_dataset = dataset_loader.load_generator_dataset("val", batch_size=args.generator_mle_batch_size)
    num_train_batches = tf.data.experimental.cardinality(train_dataset).numpy()

    for epoch in range(args.generator_mle_epochs):
        logger.info(f"-- Epoch: {epoch + 1}/{args.generator_mle_epochs}")
        for step, train_batch in enumerate(tqdm(train_dataset, total=num_train_batches, desc="Batch", unit="batch")):
            global_step.assign_add(1)
            loss, gradients = generator_mle_train_batch(encoder, generator, loss_fn,
                                                        train_batch, args.generator_mle_dsa_lambda)
            optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + generator.trainable_variables))
            if global_step % args.generator_mle_logging_steps == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("generator_mle_loss", loss, step=global_step)
            # TODO: Add global_step == args.generator_mle_logging_steps below?
            if global_step % args.generator_mle_validate_steps == 0:
                logger.info("-- Calculating validation loss")
                losses = [generator_mle_val_loss(encoder, generator, loss_fn, val_batch, args.generator_mle_dsa_lambda)
                          for val_batch in val_dataset]
                with val_summary_writer.as_default():
                    tf.summary.scalar("generator_mle_loss", tf.reduce_mean(losses), step=global_step)
            if global_step % args.generator_mle_checkpoint_steps == 0:
                checkpoint_manager.save(["encoder", "generator", "generator_mle_params"], checkpoint_number=global_step)

    logger.info("***** Training Generator using MLE - Ended *****")


def discriminator_mle_train(args, dataset_loader):
    logger.info("***** Training Discriminator using MLE - Starting *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    encoder = Encoder()
    generator = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1])
    discriminator = Discriminator(vocab_size=dataset_loader.tokenizer.vocab_size,
                                  embedding_units=args.discriminator_embedding_units,
                                  lstm_units=args.discriminator_lstm_units)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.discriminator_mle_learning_rate,
                                         clipvalue=args.discriminator_mle_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, args.overwrite_checkpoint_dir, {
        "encoder": {"encoder": encoder},
        "generator": {"generator": generator},
        "discriminator": {"discriminator": discriminator},
        "discriminator_mle_params": {"optimizer": optimizer, "global_step": global_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    train_dataset = dataset_loader.load_discriminator_dataset("train", encoder, generator,
                                                              args.discriminator_mle_batch_size,
                                                              args.discriminator_mle_faking_batch_size,
                                                              args.discriminator_mle_neg_sample_weight)
    val_dataset = dataset_loader.load_discriminator_dataset("val", encoder, generator,
                                                            args.discriminator_mle_batch_size,
                                                            args.discriminator_mle_faking_batch_size,
                                                            args.discriminator_mle_neg_sample_weight)
    num_train_batches = tf.data.experimental.cardinality(train_dataset).numpy()

    for epoch in range(args.discriminator_mle_epochs):
        logger.info(f"-- Epoch: {epoch + 1}/{args.discriminator_mle_epochs}")
        for step, train_batch in enumerate(tqdm(train_dataset, total=num_train_batches, desc="Batch", unit="batch")):
            global_step.assign_add(1)
            loss, gradients = discriminator_mle_train_batch(encoder, discriminator, loss_fn, train_batch)
            optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
            if global_step % args.discriminator_mle_logging_steps == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("discriminator_mle_loss", loss, step=global_step)
            # TODO: Add global_step == args.discriminator_mle_logging_steps below?
            if global_step % args.discriminator_mle_validate_steps == 0:
                logger.info("-- Calculating validation loss")
                losses = [discriminator_mle_val_loss(encoder, discriminator, loss_fn, val_batch)
                          for val_batch in val_dataset]
                with val_summary_writer.as_default():
                    tf.summary.scalar("discriminator_mle_loss", tf.reduce_mean(losses), step=global_step)
            if global_step % args.discriminator_mle_checkpoint_steps == 0:
                checkpoint_manager.save(["discriminator", "discriminator_mle_params"], checkpoint_number=global_step)

    logger.info("***** Training Discriminator using MLE - Ended *****")
