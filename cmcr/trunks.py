import laion_clap
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import argparse
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from cmcr.ULIP.models import ULIP_models

from cmcr.type import ModalityType

CLAP_DIR = './checkpoints/laion_clap_fullset_fusion.pt'
ULIP_DIR = './checkpoints/pointbert_ULIP-2.pt'
# CLIP_DIR = 'openai/clip-vit-base-patch32'
CLIP_DIR = '/home/wangzehan/.cache/huggingface/hub/clip-vit-base-patch32'

def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=1024, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', default=True, type=bool, help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    parser.add_argument('--test_ckpt_addr', default=ULIP_DIR, help='the ckpt to test 3d zero shot')
    
    parser.add_argument('--part', default=0, type=int)

    return parser

def load_and_transform_point_cloud_data(point_path, device):

    obj  = np.load(point_path, allow_pickle=True).item()
    xyz  = obj['xyz']
    text = obj['retrieval_text'][0]
    id   = obj['id']
    group = obj['group']
    xyz = torch.from_numpy(xyz).to(device).unsqueeze_(0).float()

    return group,id,xyz,text

class Trunk:
    def __init__(self, device) -> None:
        self.ulip_extractor = get_ulip_extractor() # ulipv2 pointbert
        self.clap_extractor = get_clap_extractor() # laion_clap
        self.clip_extractor, self.clip_processor = get_clip_extractor() # openai clip
        
        self.ulip_extractor.to(device)
        self.clap_extractor.to(device)
        self.clip_extractor.to(device)
        # self.clip_processor.to(device)
        self.device = device
        self.ulip_extractor.eval()
        self.clap_extractor.eval()
        self.clip_extractor.eval()
        
        
    def extract_feature_from_input(self, input: dict):
        # for text, input is the text itself, for other modalities, input is the filename with direct
        features = {}
        features[ModalityType.VISION] = self.get_vision_feature(input[ModalityType.VISION])
        features[ModalityType.TEXT]   = self.get_text_feature(input[ModalityType.TEXT])
        features[ModalityType.AUDIO]  = self.get_audio_feature(input[ModalityType.AUDIO])
        features[ModalityType.PC]     = self.get_3d_feature(input[ModalityType.PC])
        return features
    
    def get_vision_feature(self, files: [str]) -> Tensor:
        images = []
        for f in files:
            images.append(Image.open(f))
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        image_feature = self.clip_extractor.get_image_features(**inputs)
        # print(image_feature)
        return F.normalize(image_feature, dim=-1)
        # return image_feature
    
    def get_text_feature(self, texts: [str]) -> Tensor:
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        text_feature = self.clip_extractor.get_text_features(**inputs)
        return F.normalize(text_feature, dim=-1)
        # return text_feature
    
    def get_audio_feature(self, files: [str]) -> Tensor:
        audio_feature = self.clap_extractor.get_audio_embedding_from_filelist(x = files, use_tensor=True)
        return F.normalize(audio_feature, dim=-1)
        # return audio_feature
    
    def get_3d_feature(self, files: [str]) -> Tensor:
        pcd_feature = []
        for file in files:
            _, _, xyz, _ = load_and_transform_point_cloud_data(file, device=self.device)
            pcd_feature.append(self.ulip_extractor.encode_pc(xyz))
        pcd_feature = torch.cat(pcd_feature, dim=0)
        # print(pcd_feature.shape)
        return F.normalize(pcd_feature, dim=-1)

def get_clap_extractor() -> nn.Module:
    model = laion_clap.CLAP_Module(enable_fusion=True, device='cpu')
    model.load_ckpt(CLAP_DIR)
    return model

def get_clip_extractor() -> nn.Module:
    clip_model = CLIPModel.from_pretrained(CLIP_DIR)
    processor  = CLIPProcessor.from_pretrained(CLIP_DIR)
    return clip_model, processor

def get_ulip_extractor() -> nn.Module:
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()

    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    try:
        model = getattr(ULIP_models, old_args.model)(args=args)
        # model.to(device)
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(ULIP_models, args.model)(args=args)
        # model.to(device)
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    return model