"""Frozen SpatialCLAP spatial encoder. Input [B,2,N] float32 PCM at 16 kHz.

Architecture/state names match sarulab-speech/SpatialCLAP's
feature_extractor.py, encoder.py and seldnet.py. Audio/text CLAP stacks are not
needed for the independent spatial feature validated in this project.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class FeatureExtractor(nn.Module):
    def __init__(self,n_fft=1024,hop_length=512):
        super().__init__();self.input_ch=2;self.n_fft=n_fft;self.hop_length=hop_length
        self.register_buffer('window',torch.hann_window(n_fft))
    def forward(self,x):
        b,c,n=x.shape
        s=torch.stft(x.reshape(b*c,n),n_fft=self.n_fft,hop_length=self.hop_length,
                     window=self.window,return_complex=True)
        s=s.reshape(b,c,*s.shape[-2:])
        return torch.cat([s.abs().permute(0,3,2,1),s.angle().permute(0,3,2,1)],dim=-1)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();layers=[]
        for incoming in [4,64,64]:
            layers += [nn.Conv2d(incoming,64,kernel_size=3,padding=1),nn.ReLU(),
                       nn.BatchNorm2d(64),nn.MaxPool2d(kernel_size=(1,2))]
        self.cnn_block=nn.Sequential(*layers)
        self.middle_linear=nn.Linear(64*64,128)
        self.gru=nn.GRU(128,128,num_layers=2,batch_first=True,bidirectional=True)
    def forward(self,x):
        x=self.cnn_block(x.permute(0,3,1,2))
        b,c,t,f=x.shape
        x=x.permute(0,2,1,3).contiguous().reshape(b,t,-1)
        x=self.middle_linear(x);x,_=self.gru(x)
        return x.mean(dim=1)

class SpatialTeacher(nn.Module):
    def __init__(self,checkpoint=None):
        super().__init__()
        self.feature_extractor=FeatureExtractor();self.encoder=Encoder()
        if checkpoint is None:checkpoint=Path(__file__).resolve().parent/'weights/spatial_encoder.pt'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        state=payload.get('model_state_dict',payload)
        prefix='audio_encoder.spatial_encoder.'
        if any(k.removeprefix('module.').startswith(prefix) for k in state):
            state={k.removeprefix('module.')[len(prefix):]:v for k,v in state.items()
                   if k.removeprefix('module.').startswith(prefix)}
        self.load_state_dict(state,strict=True)
        self.requires_grad_(False);self.eval()
    def train(self,mode=True):
        # A frozen teacher must keep BatchNorm in inference mode.
        return super().train(False)
    @torch.no_grad()
    def forward(self,waveform,sample_rate=16000,normalize=True):
        if sample_rate!=16000:raise ValueError('Resample audio to 16000 Hz before teacher inference.')
        if waveform.ndim!=3 or waveform.shape[1]!=2 or waveform.shape[-1]<=512:
            raise ValueError('Expected [batch,2,samples] with more than 512 samples.')
        if not torch.isfinite(waveform).all():raise ValueError('Non-finite PCM input.')
        # Match the teacher tests: float32 forward, independent of DiT autocast.
        device=self.feature_extractor.window.device
        with torch.autocast(device_type=device.type,enabled=False):
            x=waveform.to(device=device,dtype=torch.float32)
            z=self.encoder(self.feature_extractor(x))
            return F.normalize(z,dim=-1) if normalize else z
