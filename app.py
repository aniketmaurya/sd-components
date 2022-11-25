import lightning as L

from sd_components import InpaintingServe

app = L.LightningApp(InpaintingServe(cloud_compute=L.CloudCompute("gpu-fast", disk_size=30)))
