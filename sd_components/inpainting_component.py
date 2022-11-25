import lightning as L

from sd_components.weights_downloader import ModelDownloadWork


class ModelBuildConfig(L.BuildConfig):
    def build_commands(self):
        return [
            "rm -rf stablediffusion",
            "git clone -b lit https://github.com/aniketmaurya/stablediffusion",
            "python -m pip install -r stablediffusion/requirements.txt",
            "python -m pip install -e stablediffusion",
            "pip uninstall -y opencv-python",
            "pip uninstall -y opencv-python-headless",
            "pip install opencv-python-headless==4.5.5.64",
        ]

class InpaintingServe(ModelDownloadWork):
    def __init__(self, *args, **kwargs):
        super().__init__(cloud_build_config=ModelBuildConfig(), download_repo=False, *args, **kwargs)

    def run(self, *args, **kwargs):
        config_path = "stablediffusion/configs/stable-diffusion/v2-inpainting-inference.yaml"
        url = "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt"
        super().run(config_path=config_path, weights_url=url)

        from sd_components.gradio.inpainting import launch
        launch(config=config_path, ckpt=self.weights_path, host=self.host, port=self.port)
