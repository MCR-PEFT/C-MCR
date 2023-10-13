from types import SimpleNamespace

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    PC="pointcloud",
)

MCRType = SimpleNamespace(
    CLIP="clip",
    CLAP="clap",
    ULIP="ulip"
)