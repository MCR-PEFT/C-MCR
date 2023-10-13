from cmcr.cmcr_model import C_MCR_CLAPCLIP, C_MCR_ULIPCLIP
from cmcr.cmcr_model import ModalityType, MCRType
import torch

input = {ModalityType.VISION: ['assets/toilet.jpeg',
                               'assets/BBQ.jpeg',
                               'assets/train.jpeg'],
         ModalityType.TEXT: ['a toilet',
                             'BBQ',
                             'a train'],
         ModalityType.AUDIO:['assets/toilet.wav',
                             'assets/BBQ.wav',
                             'assets/train.wav'],
         ModalityType.PC:['assets/toilet.npy',
                          'assets/BBQ.npy',
                          'assets/train.npy']
         }

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
clap_clip_model = C_MCR_CLAPCLIP(device=device)
ulip_clip_model = C_MCR_ULIPCLIP(device=device)

# you can get single modality embeddings by using these functions
# replace model by clap_clip_model or ulip_clip_model

# v_emb = model.get_vision_embedding(input)
# t_emb = model.get_text_embedding(input)
# a_emb = model.get_audio_embedding(input)
# p_emb = model.get_3d_embedding(input)

clap_clip_embeddings = clap_clip_model.get_embeddings(input)
ulip_clip_embeddings = ulip_clip_model.get_embeddings(input)

print('CLAP_CLIP')
print(
    "Vision x Text:\n",
    torch.softmax(clap_clip_embeddings[ModalityType.VISION] @ clap_clip_embeddings[ModalityType.TEXT].T * 10.0, dim=-1)
)
print(
    "Audio x Text:\n",
    torch.softmax(clap_clip_embeddings[ModalityType.AUDIO] @ clap_clip_embeddings[ModalityType.TEXT].T * 10.0, dim=-1)
)
print(
    "Audio x Vision:\n",
    torch.softmax(clap_clip_embeddings[ModalityType.AUDIO] @ clap_clip_embeddings[ModalityType.VISION].T * 10.0, dim=-1)
)

print('ULIP_CLIP')
print(
    "Vision x Text:\n",
    torch.softmax(ulip_clip_embeddings[ModalityType.VISION] @ ulip_clip_embeddings[ModalityType.TEXT].T * 10.0, dim=-1)
)
print(
    "3D x VISION:\n",
    torch.softmax(ulip_clip_embeddings[ModalityType.PC] @ ulip_clip_embeddings[ModalityType.VISION].T * 10.0, dim=-1)
)
print(
    "3D x Text:\n",
    torch.softmax(ulip_clip_embeddings[ModalityType.PC] @ ulip_clip_embeddings[ModalityType.TEXT].T * 10.0, dim=-1)
)


# Expected output

# CLAP_CLIP
# Vision x Text:
#  tensor([[0.9681, 0.0219, 0.0100],
#         [0.0403, 0.9398, 0.0199],
#         [0.0045, 0.0044, 0.9910]], device='cuda:0')
# Audio x Text:
#  tensor([[0.9937, 0.0028, 0.0035],
#         [0.0337, 0.9434, 0.0229],
#         [0.0813, 0.0253, 0.8934]], device='cuda:0')
# Audio x Vision:
#  tensor([[0.9712, 0.0079, 0.0208],
#         [0.0628, 0.8966, 0.0406],
#         [0.0322, 0.0062, 0.9616]], device='cuda:0')
# ULIP_CLIP
# Vision x Text:
#  tensor([[0.7340, 0.1593, 0.1067],
#         [0.2001, 0.5995, 0.2005],
#         [0.1622, 0.1835, 0.6542]], device='cuda:0')
# 3D x VISION:
#  tensor([[0.6783, 0.1781, 0.1436],
#         [0.1193, 0.7866, 0.0941],
#         [0.1319, 0.2170, 0.6512]], device='cuda:0')
# 3D x Text:
#  tensor([[0.7448, 0.1236, 0.1316],
#         [0.2459, 0.5625, 0.1916],
#         [0.2007, 0.2648, 0.5346]], device='cuda:0')