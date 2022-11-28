import os
import urllib.request
from dataclasses import dataclass
from os.path import basename

import lightning as L  # noqa: E402
from lightning.app.storage import Path


@dataclass
class DownloadConfig(L.BuildConfig):
    def build_commands(self):
        return [
            "rm -rf stablediffusion",
            "git clone -b lit https://github.com/aniketmaurya/stablediffusion",
            "python -m pip install -r stablediffusion/requirements.txt",
            "python -m pip install -e stablediffusion",
        ]


class ModelDownloadWork(L.LightningWork):
    """The StableDiffusionServer handles the prediction.

    It initializes a model and expose an API to handle incoming requests and generate predictions.
    """

    def __init__(self, download_repo=True, *args, **kwargs):
        if download_repo:
            super().__init__(cloud_build_config=DownloadConfig(), *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.weights_path = None
        self.config_path = None

    def download_weights(self, weights_url, config_path):
        weights_folder = Path("resources/stable_diffusion_weights")
        weights_folder.mkdir(parents=True, exist_ok=True)
        # url = "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt"
        # config_path = "stablediffusion/configs/stable-diffusion/v2-inference-v.yaml"
        weight_filename = basename(weights_url)
        weights_path = weight_filename  # "sd-weights.ckpt"
        if os.path.exists(weights_path):
            print("model weights already exists")
        else:
            print("downloading model weights...")
            urllib.request.urlretrieve(weights_url, weights_path)
            print("downloaded weights")
        self.weights_path = weights_path
        self.config_path = config_path

    def run(self, weights_url, config_path):
        self.download_weights(weights_url, config_path)
