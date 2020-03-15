import logging
import os

import tensorflow as tf
from tqdm import tqdm

from .models import Generator, Encoder, Discriminator
from .utils import set_seed, MultiCheckpointManager

logger = logging.getLogger(__name__)


class PolicyGradientLoss:
    def __call__(self, captions, predictions, rewards):
        predictions = tf.reshape(predictions, shape=(-1, predictions.shape[-1]))
        captions = tf.reshape(captions, shape=(-1,))
        rewards = tf.reshape(rewards, shape=(-1,))
        indices = tf.stack([tf.range(captions.shape[0]), captions], axis=1)
        predictions = tf.gather_nd(predictions, indices)
        loss = -tf.reduce_sum(tf.math.log(predictions) * rewards)
        return loss


class MonteCarloRollout:
    def __init__(self, generator, n_rollouts, update_rate):
        self.generator = generator
        self.update_rate = update_rate
        self.n_rollouts = n_rollouts

    def update_weights(self, generator):
        # TODO: how to incorporate update rate here?
        self.generator.set_weights(generator.get_weights())

    def calculate_rewards(self, encoder_output, predictions, discriminator):
        sequence_length = predictions.shape[1]
        rewards = []
        for t in range(1, sequence_length + 1):
            initial_values = predictions[:, :t, :]
            samples = self.generator.sample(encoder_output, initial_values, sequence_length, self.n_rollouts)
            rewards_t = tf.reduce_mean([discriminator(encoder_output, s) for s in samples], axis=0)
            rewards.append(rewards_t)
        return tf.squeeze(tf.stack(rewards, axis=1))


@tf.function
def generator_train_batch_pg(batch, encoder, generator, discriminator, loss_fn, rollout):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(generator.trainable_variables)
        loss = generator_loss_pg(batch, encoder, generator, discriminator, loss_fn, rollout)
        gradients = tape.gradient(loss, generator.trainable_variables)
    return loss, gradients


@tf.function
def generator_loss_pg(batch, encoder, generator, discriminator, loss_fn, rollout):
    logger.info("calculating loss")
    images, captions = batch
    encoder_output = encoder(images)
    predictions, _ = generator.forward(encoder_output, captions)
    rewards = rollout.calculate_rewards(encoder_output, predictions, discriminator)
    loss = loss_fn(captions, predictions, rewards)
    return loss


@tf.function
def generator_train_batch_mle(batch, encoder, generator, loss_fn, dsa_lambda):
    with tf.GradientTape() as tape:
        # TODO: calling another tf.function causes warnings. why??
        loss = generator_loss_mle(batch, encoder, generator, loss_fn, dsa_lambda)
        gradients = tape.gradient(loss, generator.trainable_variables)
    return loss, gradients


@tf.function
def generator_loss_mle(batch, encoder, generator, loss_fn, dsa_lambda):
    images, captions = batch
    encoder_output = encoder(images)
    predictions, attention_alphas = generator.forward(encoder_output, captions)
    loss = loss_fn(captions, predictions)
    # TODO: Include doubly stochastic attention loss?
    # loss += dsa_lambda * tf.reduce_mean(1. - tf.reduce_sum(attention_alphas, axis=1) ** 2)
    return loss


@tf.function
def discriminator_train_batch_mle(true_batch, fake_batch, shuffled_batch, encoder, discriminator, loss_fn):
    with tf.GradientTape() as tape:
        # TODO: calling another tf.function causes warnings. why??
        loss = discriminator_loss_mle(true_batch, encoder, discriminator, loss_fn)
        loss += discriminator_loss_mle(fake_batch, encoder, discriminator, loss_fn)
        loss += discriminator_loss_mle(shuffled_batch, encoder, discriminator, loss_fn)
        # TODO: loss /= 3 here?
        gradients = tape.gradient(loss, discriminator.trainable_variables)
    return loss, gradients


@tf.function
def discriminator_loss_mle(batch, encoder, discriminator, loss_fn):
    images, captions, labels, sample_weight = batch
    encoder_output = encoder(images)
    predictions = discriminator(encoder_output, captions)
    loss = loss_fn(labels, predictions, sample_weight=sample_weight)
    return loss


@tf.function
def generate_fake_captions(true_batch, encoder, generator, tokenizer, neg_sample_weight):
    images = true_batch[0]
    batch_size = images.shape[0]
    encoder_outputs = encoder(images)
    captions = generator.generate_caption(encoder_outputs, mode="deterministic", start_id=tokenizer.start_id,
                                          end_id=tokenizer.end_id)
    return images, captions, tf.zeros((batch_size, 1)), tf.ones((batch_size, 1)) * neg_sample_weight


def pretrain_generator(args, dataset_loader):
    logger.info("***** Pretraining Generator - Started *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    encoder = Encoder()
    generator = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.generator_pretrain_learning_rate,
                                         clipvalue=args.generator_pretrain_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, args.overwrite_checkpoint_dir, {
        "encoder": {"encoder": encoder},
        "generator": {"generator": generator},
        "generator_pretrain_params": {"optimizer": optimizer, "global_step": global_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    train_dataset = dataset_loader.load_generator_dataset(
        "train", batch_size=args.generator_pretrain_batch_size, repeat=args.generator_pretrain_epochs)
    val_dataset = dataset_loader.load_generator_dataset(
        "val", batch_size=args.generator_pretrain_batch_size, repeat=1)

    for train_batch in tqdm(train_dataset, desc="Batch", unit="batch"):
        global_step.assign_add(1)
        loss, gradients = generator_train_batch_mle(train_batch, encoder, generator, loss_fn,
                                                    args.generator_pretrain_dsa_lambda)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        if global_step % args.generator_pretrain_logging_steps == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar("generator_pretrain_loss", loss, step=global_step)
        # TODO: Calculate validation loss initially?
        if global_step % args.generator_pretrain_validate_steps == 0:
            logger.info("-- Calculating validation loss")
            losses = [generator_loss_mle(val_batch, encoder, generator, loss_fn, args.generator_pretrain_dsa_lambda)
                      for val_batch in val_dataset]
            with val_summary_writer.as_default():
                tf.summary.scalar("generator_pretrain_loss", tf.reduce_mean(losses), step=global_step)
        if global_step % args.generator_pretrain_checkpoint_steps == 0:
            checkpoint_manager.save(["encoder", "generator", "generator_pretrain_params"],
                                    checkpoint_number=global_step)

    logger.info("***** Pretraining Generator - Ended *****")


def pretrain_discriminator(args, dataset_loader):
    logger.info("***** Pretraining Discriminator - Started *****")

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.discriminator_pretrain_learning_rate,
                                         clipvalue=args.discriminator_pretrain_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, args.overwrite_checkpoint_dir, {
        "encoder": {"encoder": encoder},
        "generator": {"generator": generator},
        "discriminator": {"discriminator": discriminator},
        "discriminator_pretrain_params": {"optimizer": optimizer, "global_step": global_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    train_true_dataset, train_shuffled_dataset = dataset_loader.load_discriminator_dataset(
        "train", batch_size=args.discriminator_pretrain_batch_size,
        neg_sample_weight=args.discriminator_pretrain_neg_sample_weight, repeat=args.discriminator_pretrain_epochs)
    val_true_dataset, val_shuffled_dataset = dataset_loader.load_discriminator_dataset(
        "val", batch_size=args.discriminator_pretrain_batch_size,
        neg_sample_weight=args.discriminator_pretrain_neg_sample_weight, repeat=1)

    for true_batch, shuffled_batch in tqdm(zip(train_true_dataset, train_shuffled_dataset), desc="Batch", unit="batch"):
        global_step.assign_add(1)
        fake_batch = generate_fake_captions(true_batch, encoder, generator, dataset_loader.tokenizer,
                                            args.discriminator_pretrain_neg_sample_weight)
        loss, gradients = discriminator_train_batch_mle(true_batch, fake_batch, shuffled_batch,
                                                        encoder, discriminator, loss_fn)
        optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
        if global_step % args.discriminator_pretrain_logging_steps == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar("discriminator_pretrain_loss", loss, step=global_step)
        # TODO: Calculate validation loss initially?
        if global_step % args.discriminator_pretrain_validate_steps == 0:
            logger.info("-- Calculating validation loss")
            losses = []
            for val_true_batch, val_shuffled_batch in zip(val_true_dataset, val_shuffled_dataset):
                val_fake_batch = generate_fake_captions(val_true_batch, encoder, generator, dataset_loader.tokenizer,
                                                        args.discriminator_pretrain_neg_sample_weight)
                losses.append(
                    discriminator_loss_mle(val_true_batch, encoder, discriminator, loss_fn) +
                    discriminator_loss_mle(val_fake_batch, encoder, discriminator, loss_fn) +
                    discriminator_loss_mle(val_shuffled_batch, encoder, discriminator, loss_fn)
                )
            with val_summary_writer.as_default():
                tf.summary.scalar("discriminator_pretrain_loss", tf.reduce_mean(losses), step=global_step)
        if global_step % args.discriminator_pretrain_checkpoint_steps == 0:
            checkpoint_manager.save(["discriminator", "discriminator_pretrain_params"],
                                    checkpoint_number=global_step)

    logger.info("***** Pretraining Discriminator - Ended *****")


def adversarially_train_generator_and_discriminator(args, dataset_loader):
    logger.info("***** Adversarially training Generator & Discriminator - Started *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    encoder = Encoder()
    generator = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1])
    generator_mc = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                             embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                             attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1])
    discriminator = Discriminator(vocab_size=dataset_loader.tokenizer.vocab_size,
                                  embedding_units=args.discriminator_embedding_units,
                                  lstm_units=args.discriminator_lstm_units)
    rollout = MonteCarloRollout(generator_mc, args.adversarial_rollout_n, args.adversarial_rollout_update_rate)

    generator_loss_fn = PolicyGradientLoss()
    discriminator_loss_fn = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.generator_adversarial_learning_rate,
                                                   clipvalue=args.generator_adversarial_grad_clipvalue)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.discriminator_adversarial_learning_rate,
                                                       clipvalue=args.discriminator_adversarial_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    generator_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    discriminator_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, args.overwrite_checkpoint_dir, {
        "encoder": {"encoder": encoder},
        "generator": {"generator": generator},
        "discriminator": {"discriminator": discriminator},
        "adversarial_params": {"generator_optimizer": generator_optimizer,
                               "discriminator_optimizer": discriminator_optimizer,
                               "generator_step": generator_step, "discriminator_step": discriminator_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    generator_train_dataset = iter(dataset_loader.load_generator_dataset(
        "train", batch_size=args.generator_adversarial_batch_size, repeat=-1))
    generator_val_dataset = dataset_loader.load_generator_dataset(
        "val", batch_size=args.generator_adversarial_batch_size, repeat=1)
    discriminator_train_true_dataset, discriminator_train_shuffled_dataset = dataset_loader.load_discriminator_dataset(
        "train", batch_size=args.discriminator_adversarial_batch_size,
        neg_sample_weight=args.discriminator_adversarial_neg_sample_weight, repeat=-1)
    discriminator_val_true_dataset, discriminator_val_shuffled_dataset = dataset_loader.load_discriminator_dataset(
        "val", batch_size=args.discriminator_adversarial_batch_size,
        neg_sample_weight=args.discriminator_adversarial_neg_sample_weight, repeat=1)
    discriminator_train_true_dataset = iter(discriminator_train_true_dataset)
    discriminator_train_shuffled_dataset = iter(discriminator_train_shuffled_dataset)

    for round_ in range(args.adversarial_rounds):
        logger.info(f"-- Round: {round_ + 1}/{args.adversarial_rounds}")

        for _ in tqdm(range(args.adversarial_g_steps), desc="Training Generator", unit="batch"):
            generator_step.assign_add(1)
            train_batch = next(generator_train_dataset)
            loss, gradients = generator_train_batch_pg(train_batch, encoder, generator, discriminator,
                                                       generator_loss_fn, rollout)
            generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
            if generator_step % args.generator_adversarial_logging_steps == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("generator_adversarial_loss", loss, step=generator_step)

        for _ in tqdm(range(args.adversarial_d_steps), desc="Training Discriminator", unit="batch"):
            discriminator_step.assign_add(1)
            true_batch = next(discriminator_train_true_dataset)
            shuffled_batch = next(discriminator_train_shuffled_dataset)
            fake_batch = generate_fake_captions(true_batch, encoder, generator, dataset_loader.tokenizer,
                                                args.discriminator_adversarial_neg_sample_weight)
            loss, gradients = discriminator_train_batch_mle(true_batch, fake_batch, shuffled_batch,
                                                            encoder, discriminator, discriminator_loss_fn)
            discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
            if discriminator_step % args.discriminator_adversarial_logging_steps == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("discriminator_adversarial_loss", loss, step=discriminator_step)

        rollout.update_weights(generator)

        if round_ % args.adversarial_checkpoint_rounds:
            checkpoint_manager.save(["encoder", "generator", "discriminator", "adversarial_params"],
                                    checkpoint_number=round_)

        # TODO: Calculate validation loss initially?
        if round_ % args.adversarial_validate_rounds == 0:
            logger.info("-- Calculating generator validation loss")
            generator_losses = [
                generator_loss_pg(val_batch, encoder, generator, discriminator, generator_loss_fn, rollout)
                for val_batch in generator_val_dataset]
            logger.info("-- Calculating discriminator validation loss")
            discriminator_losses = []
            for val_true_batch, val_shuffled_batch in zip(discriminator_val_true_dataset,
                                                          discriminator_val_shuffled_dataset):
                val_fake_batch = generate_fake_captions(val_true_batch, encoder, generator, dataset_loader.tokenizer,
                                                        args.discriminator_adversarial_neg_sample_weight)
                discriminator_losses.append(
                    discriminator_loss_mle(val_true_batch, encoder, discriminator, discriminator_loss_fn) +
                    discriminator_loss_mle(val_fake_batch, encoder, discriminator, discriminator_loss_fn) +
                    discriminator_loss_mle(val_shuffled_batch, encoder, discriminator, discriminator_loss_fn)
                )
            with val_summary_writer.as_default():
                tf.summary.scalar("generator_adversarial_loss", tf.reduce_mean(generator_losses), step=generator_step)
                tf.summary.scalar("discriminator_adversarial_loss", tf.reduce_mean(discriminator_losses),
                                  step=discriminator_step)

    logger.info("***** Adversarially training Generator & Discriminator - Ended *****")
