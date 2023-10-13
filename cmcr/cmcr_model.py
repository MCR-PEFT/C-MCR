import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from cmcr.cmcr_projector import CLAPCLIP_Head, ULIPCLIP_Head
from cmcr.trunks import Trunk

from cmcr.type import ModalityType, MCRType

CLAP_CLIP = 'checkpoints/clap_clip.pt'
ULIP_CLIP = 'checkpoints/ulip_clip.pt'

class C_MCR_CLAPCLIP():
    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.device = device
        
        self.trunk = Trunk(device) # dict
        self.cmcr_head = CLAPCLIP_Head()
        self.cmcr_head.load_state_dict(torch.load(CLAP_CLIP, map_location='cpu'))
        self.cmcr_head.to(device)
        self.cmcr_head.eval()
    
    @torch.no_grad()
    def project_features(self, features: dict) -> dict:
        cmcr_embeddings = {}
        cmcr_embeddings[ModalityType.VISION] = self.project_clip(features[ModalityType.VISION])
        cmcr_embeddings[ModalityType.TEXT]   = self.project_clip(features[ModalityType.TEXT])
        cmcr_embeddings[ModalityType.AUDIO]  = self.project_clap(features[ModalityType.AUDIO])

        cmcr_embeddings[ModalityType.VISION] = F.normalize(cmcr_embeddings[ModalityType.VISION], dim=-1)
        cmcr_embeddings[ModalityType.TEXT]   = F.normalize(cmcr_embeddings[ModalityType.TEXT], dim=-1)
        cmcr_embeddings[ModalityType.AUDIO]  = F.normalize(cmcr_embeddings[ModalityType.AUDIO], dim=-1)
        
        return cmcr_embeddings
    
    @torch.no_grad()
    def project_clap(self, clap_emb: Tensor) -> Tensor:
        return self.cmcr_head.Head_A(clap_emb)
    
    @torch.no_grad()
    def project_clip(self, clip_emb: Tensor) -> Tensor:
        return self.cmcr_head.Head_B(clip_emb)
    
    @torch.no_grad()
    def get_embeddings(self, input: dict) -> dict:
        features = {}
        features[ModalityType.VISION] = self.trunk.get_vision_feature(input[ModalityType.VISION])
        features[ModalityType.TEXT]   = self.trunk.get_text_feature(input[ModalityType.TEXT])
        features[ModalityType.AUDIO]  = self.trunk.get_audio_feature(input[ModalityType.AUDIO])
        cmcr_embeddings = self.project_features(features)
        return cmcr_embeddings
    
    @torch.no_grad()
    def get_vision_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_vision_feature(input[ModalityType.VISION])
        features = self.project_clip(features)
        return F.normalize(features, dim=-1)
    
    @torch.no_grad()
    def get_text_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_text_feature(input[ModalityType.TEXT])
        features = self.project_clip(features)
        return F.normalize(features, dim=-1)
    
    @torch.no_grad()
    def get_audio_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_audio_feature(input[ModalityType.AUDIO])
        features = self.project_clap(features)
        return F.normalize(features, dim=-1)


class C_MCR_ULIPCLIP():
    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.device = device

        self.trunk = Trunk(device) # dict
        self.cmcr_head = ULIPCLIP_Head()
        self.cmcr_head.load_state_dict(torch.load(ULIP_CLIP, map_location='cpu'))
        self.cmcr_head.to(device)
        self.cmcr_head.eval()
    
    @torch.no_grad()
    def project_features(self, features: dict) -> dict:
        cmcr_embeddings = {}
        cmcr_embeddings[ModalityType.VISION] = self.project_clip(features[ModalityType.VISION])
        cmcr_embeddings[ModalityType.TEXT]   = self.project_clip(features[ModalityType.TEXT])
        cmcr_embeddings[ModalityType.PC]     = self.project_ulip(features[ModalityType.PC])

        cmcr_embeddings[ModalityType.VISION] = F.normalize(cmcr_embeddings[ModalityType.VISION], dim=-1)
        cmcr_embeddings[ModalityType.TEXT]   = F.normalize(cmcr_embeddings[ModalityType.TEXT], dim=-1)
        cmcr_embeddings[ModalityType.PC]     = F.normalize(cmcr_embeddings[ModalityType.PC], dim=-1)
        
        return cmcr_embeddings
    
    @torch.no_grad()
    def project_ulip(self, ulip_emb: Tensor) -> Tensor:
        return self.cmcr_head.Head_B(ulip_emb)
    
    @torch.no_grad()
    def project_clip(self, clip_emb: Tensor) -> Tensor:
        return self.cmcr_head.Head_A(clip_emb)
    
    @torch.no_grad()
    def get_embeddings(self, input: dict) -> dict:
        features = {}
        features[ModalityType.VISION] = self.trunk.get_vision_feature(input[ModalityType.VISION])
        features[ModalityType.TEXT]   = self.trunk.get_text_feature(input[ModalityType.TEXT])
        features[ModalityType.PC]     = self.trunk.get_3d_feature(input[ModalityType.PC])
        cmcr_embeddings = self.project_features(features)
        return cmcr_embeddings
    
    @torch.no_grad()
    def get_vision_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_vision_feature(input[ModalityType.VISION])
        features = self.project_clip(features)
        return F.normalize(features, dim=-1)
    
    @torch.no_grad()
    def get_text_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_text_feature(input[ModalityType.TEXT])
        features = self.project_clip(features)
        return F.normalize(features, dim=-1)
    
    @torch.no_grad()
    def get_3d_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_3d_feature(input[ModalityType.PC])
        features = self.project_ulip(features)
        return F.normalize(features, dim=-1)
    
