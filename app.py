import lightning as L

from sd_components import InpaintingServe

component = InpaintingServe(cloud_compute=L.CloudCompute("gpu-fast", disk_size=30))
app = L.LightningApp(component)
