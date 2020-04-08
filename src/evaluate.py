import tensorflow as tf

from .models import Encoder, Generator
from .utils import MultiCheckpointManager


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
    style_id = int(input("Enter style ID : "))

    image = dataset_manager.load_image(image_path)
    encoder_output = encoder(tf.expand_dims(image, axis=0))
    style = tf.constant(style_id, dtype=tf.int32, shape=(1,))

    sequences, sequences_logits = generator.beam_search(encoder_output, style, sequence_length=args.max_seq_len,
                                                        beam_size=5, sos=dataset_manager.tokenizer.start_id)
    print("-- Sequences:")
    print(dataset_manager.tokenizer.sequences_to_texts(sequences))
    print("-- Sequence Probabilities:")
    print(tf.math.exp(sequences_logits))
