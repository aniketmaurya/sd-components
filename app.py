import lightning as L

from sd_components import InpaintingServe

class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = InpaintingServe(cloud_compute=L.CloudCompute("gpu-fast", disk_size=30))
    
    def run(self):
        self.work.run()

    def configure_layout(self):
        return {"name": "Inpainting", "content": self.work.url}

app = L.LightningApp(RootFlow())
