import tensorflow as tf

from .models import Encoder, Generator
from .utils import MultiCheckpointManager


def generate_captions_for_image(args, dataset_manager, image_path):
    encoder = Encoder()
    generator = Generator(vocab_size=dataset_manager.tokenizer.vocab_size, lstm_units=args.generator_lstm_units,
                          embedding_units=args.generator_embedding_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=args.generator_encoder_units,
                          z_units=args.generator_z_units)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
        "generator": {"generator": generator}
    })
    checkpoint_manager.restore_latest()

    image = dataset_manager.load_image(image_path)
    encoder_output = encoder(tf.expand_dims(image, axis=0))
    sequences, sequences_log_probs = generator.beam_search(encoder_output, sequence_length=args.max_seq_len,
                                                           beam_size=5, sos=dataset_manager.tokenizer.start_id)
    print("-- Sequences:")
    print(dataset_manager.tokenizer.sequences_to_texts(sequences))
    print("-- Sequence Probabilities:")
    print(tf.math.exp(sequences_log_probs))
