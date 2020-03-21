import logging
import os
from collections import defaultdict

import tensorflow as tf
from tqdm import tqdm

from .models import Generator, Encoder, Discriminator
from .utils import set_seed, MultiCheckpointManager

logger = logging.getLogger(__name__)


class PolicyGradientLoss:
    def __call__(self, captions, probabilities, rewards):
        probabilities = tf.reshape(probabilities, shape=(-1, probabilities.shape[-1]))
        captions = tf.reshape(captions, shape=(-1,))
        rewards = tf.reshape(rewards, shape=(-1,))
        indices = tf.stack([tf.range(captions.shape[0], dtype=tf.int64), captions], axis=1)
        probabilities = tf.gather_nd(probabilities, indices)
        loss = -tf.reduce_sum(tf.math.log(probabilities) * rewards)
        return loss


class MonteCarloRollout:
    def __init__(self, generator, n_rollouts, update_rate):
        self.generator = generator
        self.update_rate = update_rate
        self.n_rollouts = n_rollouts

    def update_weights(self, generator):
        updated_weights = []
        variable_names = [v.name for v in generator.variables]
        for name, self_w, other_w in zip(variable_names, self.generator.get_weights(), generator.get_weights()):
            if "embedding" in name:
                updated_weights.append(other_w)
            else:
                updated_weights.append(self.update_rate * other_w + (1 - self.update_rate) * self_w)
        self.generator.set_weights(updated_weights)

    def calculate_rewards(self, encoder_output, captions, discriminator, training):
        sequence_length = captions.shape[1]
        rewards = []
        for t in range(1, sequence_length):
            initial_sequence = captions[:, :t]
            samples = self.generator.sample(encoder_output, initial_sequence, sequence_length,
                                            mode="stochastic", n_samples=self.n_rollouts, training=training)[0]
            rewards_t = tf.reduce_mean([discriminator(encoder_output, s) for s in samples], axis=0)
            rewards.append(rewards_t)
        rewards.append(discriminator(encoder_output, captions))
        return tf.squeeze(tf.stack(rewards, axis=1))


class InverseSigmoidDecay(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_rate, k, name="inverse_sigmoid_decay"):
        super().__init__()
        self.initial_rate = initial_rate
        self.k = k
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name):
            decay = self.k / (self.k + tf.math.exp(step / self.k))
            return self.initial_rate * decay

    def get_config(self):
        return {
            "initial_rate": self.initial_rate,
            "k": self.k,
            "name": self.name
        }


@tf.function
def generator_train_batch_pg(batch, encoder, generator, discriminator, optimizer, loss_fn, rollout, tokenizer,
                             max_seq_len):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(generator.trainable_variables)
        images = batch[0]
        batch_size = images.shape[0]
        encoder_output = encoder(images)
        initial_sequence = tf.ones((batch_size, 1), dtype=tf.int64) * tokenizer.start_id
        captions, probabilities = generator.sample(encoder_output, initial_sequence, sequence_length=max_seq_len,
                                                   mode="stochastic", n_samples=1, training=True)
        captions, probabilities = captions[0], probabilities[0]
        rewards = rollout.calculate_rewards(encoder_output, captions, discriminator, training=True)
        loss = loss_fn(captions, probabilities, rewards)
        gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss


@tf.function
def generator_loss_pg(batch, encoder, generator, discriminator, loss_fn, rollout, tokenizer, max_seq_len):
    images = batch[0]
    batch_size = images.shape[0]
    encoder_output = encoder(images)
    initial_sequence = tf.ones((batch_size, 1), dtype=tf.int64) * tokenizer.start_id
    captions, probabilities = generator.sample(encoder_output, initial_sequence, sequence_length=max_seq_len,
                                               mode="stochastic", n_samples=1, training=False)
    captions, probabilities = captions[0], probabilities[0]
    rewards = rollout.calculate_rewards(encoder_output, captions, discriminator, training=False)
    loss = loss_fn(captions, probabilities, rewards)
    return loss


@tf.function
def generator_train_batch_mle(batch, encoder, generator, loss_fn, optimizer, dsa_lambda, teacher_forcing_rate):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(generator.trainable_variables)
        images, captions = batch
        encoder_output = encoder(images)
        predictions, attention_alphas = generator.forward(encoder_output, captions, mode="stochastic",
                                                          teacher_forcing_rate=teacher_forcing_rate, training=True)
        loss = loss_fn(captions, predictions)
        loss += dsa_lambda * tf.reduce_mean(tf.reduce_sum((1 - tf.reduce_sum(attention_alphas, axis=1)) ** 2, axis=1))
        gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss


@tf.function
def generator_loss_mle(batch, encoder, generator, loss_fn, dsa_lambda):
    images, captions = batch
    encoder_output = encoder(images)
    predictions, attention_alphas = generator.forward(encoder_output, captions, mode="stochastic",
                                                      teacher_forcing_rate=0, training=False)
    loss = loss_fn(captions, predictions)
    loss += dsa_lambda * tf.reduce_mean(tf.reduce_sum((1 - tf.reduce_sum(attention_alphas, axis=1)) ** 2, axis=1))
    return loss


@tf.function
def discriminator_train_batch_mle(true_batch, fake_batch, shuffled_batch, encoder, discriminator, loss_fn, optimizer):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(discriminator.trainable_variables)
        batches = (true_batch, fake_batch, shuffled_batch)
        images = tf.concat([b[0] for b in batches], axis=0)
        labels = tf.concat([b[2] for b in batches], axis=0)
        sample_weight = tf.concat([b[3] for b in batches], axis=0)
        captions = [b[1] for b in batches]
        max_caption_length = max([c.shape[1] for c in captions])
        captions = [tf.pad(c, paddings=tf.constant([[0, 0], [0, max_caption_length - c.shape[1]]])) for c in captions]
        captions = tf.concat(captions, axis=0)
        encoder_output = encoder(images)
        predictions = discriminator(encoder_output, captions, training=True)
        loss = loss_fn(labels, predictions, sample_weight=sample_weight)
        gradients = tape.gradient(loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return loss


@tf.function
def discriminator_loss_mle(batches, encoder, discriminator, loss_fn):
    images = tf.concat([b[0] for b in batches], axis=0)
    labels = tf.concat([b[2] for b in batches], axis=0)
    sample_weight = tf.concat([b[3] for b in batches], axis=0)
    captions = [b[1] for b in batches]
    max_caption_length = max([c.shape[1] for c in captions])
    captions = [tf.pad(c, paddings=tf.constant([[0, 0], [0, max_caption_length - c.shape[1]]])) for c in captions]
    captions = tf.concat(captions, axis=0)
    encoder_output = encoder(images)
    predictions = discriminator(encoder_output, captions)
    loss = loss_fn(labels, predictions, sample_weight=sample_weight)
    return loss


@tf.function
def generate_fake_captions(true_batch, encoder, generator, tokenizer, max_seq_len):
    images, labels, sample_weights = true_batch[0], true_batch[2], true_batch[3]
    batch_size = images.shape[0]
    encoder_output = encoder(images)
    initial_sequence = tf.ones((batch_size, 1), dtype=tf.int64) * tokenizer.start_id
    captions = generator.sample(encoder_output, initial_sequence, sequence_length=max_seq_len,
                                mode="stochastic", n_samples=1, training=False)[0][0]
    return images, captions, labels, sample_weights


def pretrain_generator(args, dataset_loader):
    logger.info("***** Pretraining Generator - Started *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    encoder = Encoder()
    generator = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1],
                          z_units=args.generator_z_units)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.generator_pretrain_learning_rate,
                                         clipvalue=args.generator_pretrain_grad_clipvalue)
    teacher_forcing_schedule = InverseSigmoidDecay(args.generator_pretrain_scheduled_sampling_initial_rate,
                                                   args.generator_pretrain_scheduled_sampling_k)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
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
        teacher_forcing_rate = teacher_forcing_schedule(global_step)
        loss = generator_train_batch_mle(train_batch, encoder, generator, loss_fn, optimizer,
                                         args.generator_pretrain_dsa_lambda, teacher_forcing_rate)
        if global_step % args.generator_pretrain_logging_steps == 0:
            with train_summary_writer.as_default(), tf.name_scope("generator_pretraining"):
                tf.summary.scalar("crossentropy_loss", loss, step=global_step)
                tf.summary.scalar("teacher_forcing_rate", teacher_forcing_rate, step=global_step)
        # TODO: Calculate validation loss initially?
        if global_step % args.generator_pretrain_validate_steps == 0:
            logger.info("-- Calculating validation loss")
            losses = [generator_loss_mle(val_batch, encoder, generator, loss_fn, args.generator_pretrain_dsa_lambda)
                      for val_batch in val_dataset]
            with val_summary_writer.as_default(), tf.name_scope("generator_pretraining"):
                tf.summary.scalar("crossentropy_loss", tf.reduce_mean(losses), step=global_step)
        if global_step % args.generator_pretrain_checkpoint_steps == 0:
            checkpoint_manager.save(["encoder", "generator", "generator_pretrain_params"])

    logger.info("***** Pretraining Generator - Ended *****")


def pretrain_discriminator(args, dataset_loader):
    logger.info("***** Pretraining Discriminator - Started *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    encoder = Encoder()
    generator = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1],
                          z_units=args.generator_z_units)
    discriminator = Discriminator(vocab_size=dataset_loader.tokenizer.vocab_size,
                                  embedding_units=args.discriminator_embedding_units,
                                  lstm_units=args.discriminator_lstm_units)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.discriminator_pretrain_learning_rate,
                                         clipvalue=args.discriminator_pretrain_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
        "encoder": {"encoder": encoder},
        "generator": {"generator": generator},
        "discriminator": {"discriminator": discriminator},
        "discriminator_pretrain_params": {"optimizer": optimizer, "global_step": global_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    train_dataset, val_dataset = {}, {}
    for s, d, r in [("train", train_dataset, args.discriminator_pretrain_epochs), ("val", val_dataset, 1)]:
        d["true"] = dataset_loader.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_pretrain_batch_size, repeat=r,
            label=1, randomize_captions=False, sample_weight=1)
        d["fake"] = dataset_loader.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_pretrain_batch_size, repeat=r,
            label=0, randomize_captions=False, sample_weight=args.discriminator_pretrain_neg_sample_weight)
        d["shuffled"] = dataset_loader.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_pretrain_batch_size, repeat=r,
            label=0, randomize_captions=True, sample_weight=args.discriminator_pretrain_neg_sample_weight)

    for true_batch, fake_batch, shuffled_batch in tqdm(zip(train_dataset["true"], train_dataset["fake"],
                                                           train_dataset["shuffled"]), desc="Batch", unit="batch"):
        global_step.assign_add(1)
        fake_batch = generate_fake_captions(fake_batch, encoder, generator, dataset_loader.tokenizer, args.max_seq_len)
        loss = discriminator_train_batch_mle(true_batch, fake_batch, shuffled_batch, encoder, discriminator, loss_fn,
                                             optimizer)
        if global_step % args.discriminator_pretrain_logging_steps == 0:
            with train_summary_writer.as_default(), tf.name_scope("discriminator_pretraining"):
                tf.summary.scalar("crossentropy_loss", loss, step=global_step)
        # TODO: Calculate validation loss initially?
        if global_step % args.discriminator_pretrain_validate_steps == 0:
            logger.info("-- Calculating validation loss")
            losses = []
            for val_true_batch, val_fake_batch, val_shuffled_batch in zip(val_dataset["true"], val_dataset["fake"],
                                                                          val_dataset["shuffled"]):
                val_fake_batch = generate_fake_captions(val_fake_batch, encoder, generator, dataset_loader.tokenizer,
                                                        args.max_seq_len)
                losses.append(discriminator_loss_mle((val_true_batch, val_fake_batch, val_shuffled_batch),
                                                     encoder, discriminator, loss_fn))
            with val_summary_writer.as_default(), tf.name_scope("discriminator_pretraining"):
                tf.summary.scalar("crossentropy_loss", tf.reduce_mean(losses), step=global_step)
        if global_step % args.discriminator_pretrain_checkpoint_steps == 0:
            checkpoint_manager.save(["discriminator", "discriminator_pretrain_params"])

    logger.info("***** Pretraining Discriminator - Ended *****")


def adversarially_train_generator_and_discriminator(args, dataset_loader):
    logger.info("***** Adversarially training Generator & Discriminator - Started *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    encoder = Encoder()
    generator = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1],
                          z_units=args.generator_z_units)
    generator_mc = Generator(vocab_size=dataset_loader.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                             embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                             attention_units=args.generator_attention_units, encoder_units=encoder.output_shape[-1],
                             z_units=args.generator_z_units)
    discriminator = Discriminator(vocab_size=dataset_loader.tokenizer.vocab_size,
                                  embedding_units=args.discriminator_embedding_units,
                                  lstm_units=args.discriminator_lstm_units)
    rollout = MonteCarloRollout(generator_mc, args.adversarial_rollout_n, args.adversarial_rollout_update_rate)
    rollout.update_weights(generator)

    generator_loss_fn_pg = PolicyGradientLoss()
    generator_loss_fn_mle = tf.keras.losses.SparseCategoricalCrossentropy()
    discriminator_loss_fn = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.generator_adversarial_learning_rate,
                                                   clipvalue=args.generator_adversarial_grad_clipvalue)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.discriminator_adversarial_learning_rate,
                                                       clipvalue=args.discriminator_adversarial_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    generator_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    discriminator_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
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
    discriminator_train_dataset, discriminator_val_dataset = {}, {}
    for s, d, r in [("train", discriminator_train_dataset, -1), ("val", discriminator_val_dataset, 1)]:
        d["true"] = dataset_loader.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_adversarial_batch_size, repeat=r,
            label=1, randomize_captions=False, sample_weight=1)
        d["fake"] = dataset_loader.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_adversarial_batch_size, repeat=r,
            label=0, randomize_captions=False, sample_weight=args.discriminator_adversarial_neg_sample_weight)
        d["shuffled"] = dataset_loader.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_adversarial_batch_size, repeat=r,
            label=0, randomize_captions=True, sample_weight=args.discriminator_adversarial_neg_sample_weight)
    discriminator_train_dataset = {k: iter(v) for k, v in discriminator_train_dataset.items()}

    for round_ in range(1, args.adversarial_rounds):
        logger.info(f"-- Round: {round_}/{args.adversarial_rounds}")

        for _ in tqdm(range(args.adversarial_g_steps), desc="Training Generator", unit="batch"):
            generator_step.assign_add(1)
            train_batch = next(generator_train_dataset)
            pg_loss = generator_train_batch_pg(train_batch, encoder, generator, discriminator, generator_optimizer,
                                               generator_loss_fn_pg, rollout, dataset_loader.tokenizer,
                                               args.max_seq_len)
            if generator_step % args.generator_adversarial_logging_steps == 0:
                mle_loss = generator_loss_mle(train_batch, encoder, generator, generator_loss_fn_mle,
                                              args.generator_adversarial_dsa_lambda)
                with train_summary_writer.as_default(), tf.name_scope("generator_adversarial_training"):
                    tf.summary.scalar("policy_gradient_loss", pg_loss, step=generator_step)
                    tf.summary.scalar("crossentropy_loss", mle_loss, step=generator_step)

        for _ in tqdm(range(args.adversarial_d_steps), desc="Training Discriminator", unit="batch"):
            discriminator_step.assign_add(1)
            true_batch = next(discriminator_train_dataset["true"])
            fake_batch = next(discriminator_train_dataset["fake"])
            shuffled_batch = next(discriminator_train_dataset["shuffled"])
            fake_batch = generate_fake_captions(fake_batch, encoder, generator, dataset_loader.tokenizer,
                                                args.max_seq_len)
            loss = discriminator_train_batch_mle(true_batch, fake_batch, shuffled_batch, encoder, discriminator,
                                                 discriminator_loss_fn, discriminator_optimizer)
            if discriminator_step % args.discriminator_adversarial_logging_steps == 0:
                with train_summary_writer.as_default(), tf.name_scope("discriminator_adversarial_training"):
                    tf.summary.scalar("crossentropy_loss", loss, step=discriminator_step)

        rollout.update_weights(generator)

        if round_ % args.adversarial_checkpoint_rounds:
            checkpoint_manager.save(["encoder", "generator", "discriminator", "adversarial_params"])

        # TODO: Calculate validation loss initially?
        if round_ % args.adversarial_validate_rounds == 0:
            logger.info("-- Calculating generator validation loss")
            generator_losses = defaultdict(list)
            for val_batch in generator_val_dataset:
                generator_losses["pg"].append(
                    generator_loss_pg(val_batch, encoder, generator, discriminator, generator_loss_fn_pg, rollout,
                                      dataset_loader.tokenizer, args.max_seq_len)
                )
                generator_losses["mle"].append(generator_loss_mle(val_batch, encoder, generator, generator_loss_fn_mle,
                                                                  args.generator_adversarial_dsa_lambda))
            with val_summary_writer.as_default(), tf.name_scope("generator_adversarial_training"):
                tf.summary.scalar("policy_gradient_loss", tf.reduce_mean(generator_losses["pg"]), step=generator_step)
                tf.summary.scalar("crossentropy_loss", tf.reduce_mean(generator_losses["mle"]), step=generator_step)

            logger.info("-- Calculating discriminator validation loss")
            discriminator_losses = []
            for val_true_batch, val_fake_batch, val_shuffled_batch in zip(discriminator_val_dataset["true"],
                                                                          discriminator_val_dataset["fake"],
                                                                          discriminator_val_dataset["shuffled"]):
                val_fake_batch = generate_fake_captions(val_fake_batch, encoder, generator, dataset_loader.tokenizer,
                                                        args.max_seq_len)
                discriminator_losses.append(discriminator_loss_mle((val_true_batch, val_fake_batch, val_shuffled_batch),
                                                                   encoder, discriminator, discriminator_loss_fn))
            with val_summary_writer.as_default(), tf.name_scope("discriminator_adversarial_training"):
                tf.summary.scalar("crossentropy_loss", tf.reduce_mean(discriminator_losses), step=discriminator_step)

    logger.info("***** Adversarially training Generator & Discriminator - Ended *****")
