import lightning as L

from weights_downloader import ModelDownloadWork

class InpaintingServe(ModelDownloadWork):
    def run(self, *args, **kwargs):
        config_path = "stablediffusion/configs/stable-diffusion/v2-inpainting-inference.yaml"
        url = "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt"
        super().run(config_path=config_path, weights_url=url)

        from sd_components.gradio.inpainting import launch
        launch(config=config_path, ckpt=self.weights_path, port=self.port)

component = InpaintingServe()
app = L.LightningApp(component)
