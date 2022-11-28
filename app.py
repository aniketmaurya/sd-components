import lightning as L

from sd_components import InpaintingServe, InpaintingStreamlitServe


# app = L.LightningApp(InpaintingServe(cloud_compute=L.CloudCompute("gpu-fast", disk_size=30)))
app = L.LightningApp(InpaintingStreamlitServe())
