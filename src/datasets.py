import json
import logging
import os
import tarfile
import urllib.request
from urllib.error import HTTPError

import io
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from preprocess import Tokenizer

logger = logging.getLogger(__name__)


class PersonalityCaptions:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, "images")
        self.image_url_prefix = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images"
        self.captions_url = "http://parl.ai/downloads/personality_captions/personality_captions.tgz"
        self.train_file, self.val_file, self.test_file = "train.json", "val.json", "test.json"
        self.dataset_files = {"train": self.train_file, "val": self.val_file, "test": self.test_file}
        self.metadata_files = ["personalities.json", "personalities.txt"]

    def download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not all([cf in os.listdir(self.data_dir) for cf in self.metadata_files + list(self.dataset_files.values())]):
            response = urllib.request.urlopen(self.captions_url)
            tar = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
            tar.extractall(path=self.data_dir)
            tar.close()
        hashes = []
        for fname in self.dataset_files.values():
            with open(os.path.join(self.data_dir, fname), "r") as f:
                data = json.load(f)
                hashes += [d["image_hash"] for d in data]
        os.makedirs(self.image_dir, exist_ok=True)
        downloaded_images = set(os.listdir(self.image_dir))
        for hash in tqdm(hashes, unit="img"):
            image_fname = f"{hash}.jpg"
            if image_fname in downloaded_images:
                continue
            image_url = f"{self.image_url_prefix}/{hash[:3]}/{hash[3:6]}/{image_fname}"
            try:
                response = urllib.request.urlopen(image_url)
                with open(os.path.join(self.image_dir, image_fname), "wb") as f:
                    f.write(response.read())
                    downloaded_images.add(image_fname)
            except HTTPError as e:
                print(f"HTTP Error {e.code} - {image_url}")
                continue

    def load(self, split):
        file_to_load = self.dataset_files[split]
        downloaded_images = set(os.listdir(self.image_dir))
        with open(os.path.join(self.data_dir, file_to_load), "r") as f:
            data = json.load(f)
        data = filter(lambda d: f"{d['image_hash']}.jpg" in downloaded_images, data)
        data = map(lambda d: {
            "style": d["personality"],
            "caption": d["comment"],
            "additional_captions": d["additional_comments"] if "additional_comments" in d else [],
            "image_path": os.path.join(self.image_dir, f"{d['image_hash']}.jpg")
        }, data)
        return list(data)


class DatasetLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([d["caption"] for d in self.dataset.load("train")])

    def load_generator_dataset(self, split, batch_size):
        data = self.dataset.load(split)
        for d in data:
            d.update({"caption": self.tokenizer.texts_to_sequences([d["caption"]])[0]})
        # TODO: sort here or not? (efficiency vs. bias during training). Maybe shuffle here.
        data = list(sorted(data, key=lambda d: len(d["caption"])))
        image_paths = tf.convert_to_tensor([d["image_path"] for d in data])
        sequences = tf.ragged.constant([d["caption"] for d in data])
        tf_dataset = tf.data.Dataset.from_tensor_slices((image_paths, sequences))
        tf_dataset = tf_dataset.map(self._image_sequence_mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([299, 299, 3], [None]))
        tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return tf_dataset

    def load_discriminator_dataset(self, split, encoder, generator, batch_size, faking_batch_size, neg_sample_weight):
        true_data = self.dataset.load(split)
        # TODO: is sampling only 1/3 of the data each time enough?
        # TODO: do this for train, but not for val?
        true_data = np.random.choice(true_data, size=int(len(true_data) / 3), replace=False)
        for d in true_data:
            d.update({
                "caption": self.tokenizer.texts_to_sequences([d["caption"]])[0],
                "discriminator_label": 1,
                "sample_weight": 1
            })
        logger.info("-- Generating fake data")
        fake_captions = self._generate_fake_captions(true_data, faking_batch_size, encoder, generator)
        shuffled_captions = [d["caption"] for d in true_data]
        np.random.shuffle(shuffled_captions)
        fake_data, shuffled_data = [], []
        for d, fake_cap, shuffled_cap in zip(true_data, fake_captions, shuffled_captions):
            fake_data.append({**d, "caption": fake_cap, "discriminator_label": 0,
                              "sample_weight": neg_sample_weight})
            shuffled_data.append({**d, "caption": shuffled_cap, "discriminator_label": 0,
                                  "sample_weight": neg_sample_weight})
        data = true_data + fake_data + shuffled_data
        np.random.shuffle(data)
        image_paths = tf.convert_to_tensor([d["image_path"] for d in data])
        sequences = tf.ragged.constant([d["caption"] for d in data])
        discriminator_labels = tf.convert_to_tensor([[d["discriminator_label"]] for d in data])
        sample_weights = tf.convert_to_tensor([[d["sample_weight"]] for d in data])
        tf_dataset = tf.data.Dataset.from_tensor_slices((image_paths, sequences, discriminator_labels, sample_weights))
        tf_dataset = tf_dataset.map(self._image_sequence_label_weight_mapper,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([299, 299, 3], [None], [None], [None]))
        tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return tf_dataset

    def _generate_fake_captions(self, true_data, batch_size, encoder, generator):
        image_paths = tf.convert_to_tensor([d["image_path"] for d in true_data])
        d = tf.data.Dataset.from_tensor_slices((image_paths,))
        d = d.map(self._image_mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        d = d.batch(batch_size)
        d = d.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        fake_captions = []
        num_batches = tf.data.experimental.cardinality(d).numpy()
        for image_batch in tqdm(d, total=num_batches, desc="Batch", unit="batch"):
            captions = self._generate_caption(image_batch, encoder, generator).numpy()
            captions = [c[c != self.tokenizer.pad_id].tolist() for c in captions]
            fake_captions.extend(captions)
        return fake_captions

    @tf.function
    def _generate_caption(self, image_batch, encoder, generator):
        encoder_outputs = encoder(image_batch)
        return generator.generate_caption(encoder_outputs, mode="deterministic", start_id=self.tokenizer.start_id,
                                          end_id=self.tokenizer.end_id)

    @staticmethod
    def _image_sequence_label_weight_mapper(image_path, sequence, label, weight):
        img, sequence = DatasetLoader._image_sequence_mapper(image_path, sequence)
        return img, sequence, label, weight

    @staticmethod
    def _image_sequence_mapper(image_path, sequence):
        img = DatasetLoader._image_mapper(image_path)
        return img, sequence

    @staticmethod
    def _image_mapper(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.image.resize(img, [299, 299])
        return img
