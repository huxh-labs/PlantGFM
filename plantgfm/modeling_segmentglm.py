import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_plantglm import PlantGLMForCausalLM

class SegmentGLMConfig(PretrainedConfig):
    model_type = "segmentglm"
    def __init__(
        self,
        pre_trained_path = None, 
        unet_embd_dim = [1024,1536,2560,4096], 
        unet_kernel_size = 3, 
        unet_dilation = [6,12,24], 
        unet_padding = [6,12,24], 
        unet_layer_dropout = 0.25, 
        out_embd_dim = 256, 
        out_k = 1, 
        **kwargs,
    ):
        self.pre_trained_path = pre_trained_path
        self.unet_embd_dim = unet_embd_dim
        self.unet_kernel_size = unet_kernel_size
        self.unet_dilation = unet_dilation
        self.unet_padding = unet_padding
        self.unet_layer_dropout = unet_layer_dropout
        self.out_embd_dim = out_embd_dim
        self.out_k = out_k

        super().__init__(**kwargs)

    @classmethod
    def from_original_config(cls, config_path, **kwargs):
        with open(config_path, "r") as f:
            config = json.load(f)

        pre_trained_path = config["pre_trained_path"]
        unet_embd_dim = config["unet_embd_dim"]
        unet_kernel_size = config["unet_kernel_size"]
        unet_dilation = config["unet_dilation"]
        unet_padding = config["unet_padding"]
        unet_layer_dropout = config["unet_layer_dropout"]
        out_embd_dim = config["out_embd_dim"]
        out_k = config["out_k"]

        return cls(
            pre_trained_path = pre_trained_path,
            unet_embd_dim = unet_embd_dim,
            unet_kernel_size = unet_kernel_size,
            unet_dilation = unet_dilation,
            unet_padding = unet_padding,
            unet_layer_dropout = unet_layer_dropout,
            out_embd_dim = out_embd_dim,
            out_k = out_k,
            **kwargs
        )

class PlantGLMEmbd(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        glm_model = PlantGLMForCausalLM.from_pretrained(self.config.pre_trained_path)
        self.glm_decoder = glm_model.get_decoder()

    def forward(self, input_ids):
        embd = self.glm_decoder(input_ids, return_dict=True)["last_hidden_state"]
        embd = embd[:,1:-1,:].transpose(1,2)
        return embd
    
class DilatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, dropout_rate):
        super().__init__()
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=padding, 
                dilation=dilation, 
            ),
            nn.Conv1d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=padding, 
                dilation=dilation, 
            ),
            nn.SiLU(),
            nn.Dropout1d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor):

        return self.dilated_conv(x)

    
class DilatedUNetHead(nn.Module):
    def __init__(
            self, 
            embd_dim,
            kernel_size,
            padding,
            dilation,
            layer_dropout=0.25,
            out_embd_dim=256,
            out_k=1,
        ):
        super().__init__()
        self.out_k = out_k
        self.down_conv1 = DilatedConvLayer(embd_dim[0], embd_dim[1], kernel_size, padding[0], dilation[0], layer_dropout)
        self.down_conv2 = DilatedConvLayer(embd_dim[1], embd_dim[2], kernel_size, padding[1], dilation[1], layer_dropout)
        self.down_conv3 = DilatedConvLayer(embd_dim[2], embd_dim[3], kernel_size, padding[2], dilation[2], layer_dropout)

        self.up_trans1 = nn.ConvTranspose1d(embd_dim[3], embd_dim[2], kernel_size=2, stride=2, groups=64)
        self.up_trans2 = nn.ConvTranspose1d(embd_dim[2], embd_dim[1], kernel_size=2, stride=2, groups=64)

        self.up_conv1 = DilatedConvLayer(2*embd_dim[2], embd_dim[2], kernel_size, padding[1], dilation[1], layer_dropout)
        self.up_conv2 = DilatedConvLayer(2*embd_dim[1], out_embd_dim, kernel_size, padding[0], dilation[0], layer_dropout)

        self.output = nn.Conv1d(in_channels=out_embd_dim, out_channels=out_k, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor): 
        x = self.down_conv1(x) 
        t1 = x

        x = F.avg_pool1d(x, kernel_size=2, stride=2) 
        x = self.down_conv2(x) 
        t3 = x

        x = F.avg_pool1d(x, kernel_size=2, stride=2) 
        x = self.down_conv3(x) 


        x = self.up_trans1(x) 
        x = torch.cat([x, t3], 1)
        x = self.up_conv1(x) 

        x = self.up_trans2(x) 
        x = torch.cat([x, t1], 1)
        x = self.up_conv2(x)

        x = self.output(x)

        if self.out_k == 1:
            return x.squeeze(1) # when out_k==1 return target (bsz, L)
        
        return x # return target (bsz, out_k, L)

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):

        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)  # (batch_size, *)
        targets = targets.view(targets.size(0), -1)  # (batch_size, *)

        intersection = (inputs * targets).sum(dim=1)  
        total = (inputs + targets).sum(dim=1)
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6, bce_weight=0.5, iou_weight=0.5):

        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, inputs, targets):

        bce = self.bce_loss(inputs, targets)
        iou = self.iou_loss(inputs, targets)
        combined_loss = self.bce_weight * bce + self.iou_weight * iou
        return combined_loss

class SegmentGLMModel(PreTrainedModel):
    config_class = SegmentGLMConfig
    _no_split_modules = ["DilatedUNetHead"]
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.glm_embd = PlantGLMEmbd(config=config)
        self.unet_head = DilatedUNetHead(
            self.config.unet_embd_dim,
            self.config.unet_kernel_size,
            self.config.unet_padding,
            self.config.unet_dilation,
            self.config.unet_layer_dropout,
            self.config.out_embd_dim,
            self.config.out_k
        )

        self.loss_funct = CombinedLoss(bce_weight=0.5, iou_weight=0.5)

    def forward(self, input_ids: torch.LongTensor = None, labels: Optional[torch.FloatTensor] = None):
        x = self.glm_embd(input_ids)
        x = self.unet_head(x)
        if labels is None:
            return x
        
        return {
            "loss": self.loss_funct(x, labels),
            "predictions": x
        }
