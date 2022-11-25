import lightning as L

from sd_components import InpaintingServe

component = InpaintingServe()
app = L.LightningApp(component)
