import io
import json
import os
import tarfile
import urllib.request
from urllib.error import HTTPError

from tqdm import tqdm


class PersonalityCaptions:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, "images")
        self.image_url_prefix = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images"
        self.captions_url = "http://parl.ai/downloads/personality_captions/personality_captions.tgz"
        self.metadata_files = ["personalities.json", "personalities.txt"]
        self.dataset_files = ["train.json", "val.json", "test.json"]

    def download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not all([cf in os.listdir(self.data_dir) for cf in self.metadata_files + self.dataset_files]):
            response = urllib.request.urlopen(self.captions_url)
            tar = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
            tar.extractall(path=self.data_dir)
            tar.close()
        hashes = []
        for fname in self.dataset_files:
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
