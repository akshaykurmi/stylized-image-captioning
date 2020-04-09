import logging

import tensorflow as tf

from .models import Encoder, Generator
from .utils import MultiCheckpointManager

logger = logging.getLogger(__name__)


def generate_captions_for_image(args, dataset_manager):
    encoder = Encoder()
    generator = Generator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                          style_vocab_size=dataset_manager.style_encoder.num_classes,
                          style_embedding_units=args.generator_style_embedding_units,
                          token_embedding_units=args.generator_token_embedding_units,
                          lstm_units=args.generator_lstm_units,
                          lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units,
                          encoder_units=args.generator_encoder_units,
                          z_units=args.generator_z_units, stylize=args.stylize)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
        "generator": {"generator": generator}
    })
    checkpoint_manager.restore_latest()

    image_path = input("Enter image path : ")
    style = input("Enter style: ")

    image = dataset_manager.load_image(image_path)
    encoder_output = encoder(tf.expand_dims(image, axis=0))
    style = tf.constant(dataset_manager.style_encoder.label_to_index[style], dtype=tf.int32, shape=(1,))

    sequences, sequences_logits = generator.beam_search(encoder_output, style, sequence_length=args.max_seq_len,
                                                        beam_size=5, sos=dataset_manager.tokenizer.start_id,
                                                        eos=dataset_manager.tokenizer.end_id)
    logger.info("-- Beam Search")
    for seq, logit in zip(sequences.numpy()[0], sequences_logits.numpy()[0]):
        logger.info(f"Logit: {logit:0.5f} | Seq: {_seq_to_text(dataset_manager, seq)}")

    logger.info("-- Random Sampling")
    initial_sequence = tf.ones((1, 1), dtype=tf.int64) * dataset_manager.tokenizer.start_id
    sequences = generator.sample(encoder_output, initial_sequence, style,
                                 sequence_length=args.max_seq_len, mode="stochastic", n_samples=5,
                                 training=False, eos=dataset_manager.tokenizer.end_id)[0]
    for seq in sequences:
        logger.info(f"Seq: {_seq_to_text(dataset_manager, seq.numpy()[0])}")


def _seq_to_text(dataset_manager, seq):
    text = dataset_manager.tokenizer.sequences_to_texts(seq[seq > 0]).numpy()
    return " ".join([t.decode("utf-8") for t in text][1:-1])
