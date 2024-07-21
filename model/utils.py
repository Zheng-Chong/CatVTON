import os
import json
import torch
from model.attn_processor import AttnProcessor2_0, SkipAttnProcessor 


def init_adapter(unet, 
                 cross_attn_cls=SkipAttnProcessor,
                 self_attn_cls=None,
                 cross_attn_dim=None, 
                 **kwargs):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
        else:
            attn_procs[name] = cross_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
                                                    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules

def init_diffusion_model(diffusion_model_name_or_path, unet_class=None):
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    text_encoder = CLIPTextModel.from_pretrained(diffusion_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(diffusion_model_name_or_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name_or_path, subfolder="tokenizer")
    try:
        unet_folder = os.path.join(diffusion_model_name_or_path, "unet")
        unet_configs = json.load(open(os.path.join(unet_folder, "config.json"), "r"))
        unet = unet_class(**unet_configs)
        unet.load_state_dict(torch.load(os.path.join(unet_folder, "diffusion_pytorch_model.bin"), map_location="cpu"), strict=True)
    except:
        unet = None
    return text_encoder, vae, tokenizer, unet

def attn_of_unet(unet):
    attn_blocks = torch.nn.ModuleList()
    for name, param in unet.named_modules():
        if "attn1" in name:
            attn_blocks.append(param)
    return attn_blocks

def get_trainable_module(unet, trainable_module_name):
    if trainable_module_name == "unet":
        return unet
    elif trainable_module_name == "transformer":
        trainable_modules = torch.nn.ModuleList()
        for blocks in [unet.down_blocks, unet.mid_block, unet.up_blocks]:
            if hasattr(blocks, "attentions"):
                trainable_modules.append(blocks.attentions)
            else:
                for block in blocks:
                    if hasattr(block, "attentions"):
                        trainable_modules.append(block.attentions)
        return trainable_modules
    elif trainable_module_name == "attention":
        attn_blocks = torch.nn.ModuleList()
        for name, param in unet.named_modules():
            if "attn1" in name:
                attn_blocks.append(param)
        return attn_blocks
    else:
        raise ValueError(f"Unknown trainable_module_name: {trainable_module_name}")

                
    
