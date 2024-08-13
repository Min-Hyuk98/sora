# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from opensora.registry import MODELS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import (
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    to_torch_dtype,
)

import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import utils
import utils_img
import utils_model

# sys.path.append('src')
# from ldm.models.autoencoder import AutoencoderKL
# from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider
import pdb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(params):

    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    
    # Print the arguments
    # print("__git__:{}".format(utils.get_sha()))
    # print("__log__:{}".format(json.dumps(vars(params))))
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = params.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(params.get("dtype", "bf16"))

    # Create the directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    params.imgs_dir = imgs_dir
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    # Loads LDM auto-encoder models
    print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    # config = OmegaConf.load(f"{params.ldm_config}")
    # ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, params.ldm_ckpt)
    # ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    # ldm_ae.eval()
    # ldm_ae.to(device)
    
    
    vae = build_module(params.get("vae", None), MODELS)
    vae = vae.to(device, dtype).eval()

    
    # Loads hidden decoder
    print(f'>>> Building hidden decoder with weights from {params.msg_decoder_path}...')
    if 'torchscript' in params.msg_decoder_path: 
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        # already whitened
        
    else:
        msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
        ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
        msg_decoder.eval()

        # whitening
        print(f'>>> Whitening...')
        with torch.no_grad():
            # features from the dataset
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            loader = utils.get_dataloader(params.train_dir, transform, batch_size=16, collate_fn=None)
            ys = []
            for i, x in enumerate(loader):
                x = x.to(device)
                y = msg_decoder(x)
                ys.append(y.to('cpu'))
            ys = torch.cat(ys, dim=0)
            nbit = ys.shape[1]
            
            # whitening
            mean = ys.mean(dim=0, keepdim=True) # NxD -> 1xD
            ys_centered = ys - mean # NxD
            cov = ys_centered.T @ ys_centered
            e, v = torch.linalg.eigh(cov)
            L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
            weight = torch.mm(L, v.T)
            bias = -torch.mm(mean, weight.T).squeeze(0)
            linear = nn.Linear(nbit, nbit, bias=True)
            linear.weight.data = np.sqrt(nbit) * weight
            linear.bias.data = np.sqrt(nbit) * bias
            msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
            torchscript_m = torch.jit.script(msg_decoder)
            params.msg_decoder_path = params.msg_decoder_path.replace(".pth", "_whit.pth")
            print(f'>>> Creating torchscript at {params.msg_decoder_path}...')
            torch.jit.save(torchscript_m, params.msg_decoder_path)
    
    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]

    # Freeze LDM and hidden decoder
    for param in [*msg_decoder.parameters(), *vae.parameters()]:
        param.requires_grad = False

    # Loads the data
    print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size*4, num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    
    # Create losses
    print(f'>>> Creating losses...')
    print(f'Losses: {params.loss_w} and {params.loss_i}...')
    if params.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif params.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    
    if params.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    elif params.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    else:
        raise NotImplementedError

    for ii_key in range(params.num_keys):
        # Creating key
        print(f'\n>>> Creating key with {nbit} bits...')
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
        key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
        print(f'Key: {key_str}')

        # Copy the LDM decoder and finetune the copy
        ldm_decoder = deepcopy(vae)
        # ldm_decoder.encoder = nn.Identity()
        # ldm_decoder.quant_conv = nn.Identity()
        ldm_decoder.to(device)
        # for param in ldm_decoder.parameters():
        #     param.requires_grad = True
        for name, layer in ldm_decoder.named_parameters():
            if 'decoder' in name:
                layer.requires_grad = True
                # print(f"Layer Name: {name}, {layer.requires_grad}")
        optim_params = utils.parse_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)

        # Training loop
        print(f'>>> Training...')
                
        train_stats = train(train_loader, optimizer, loss_w, loss_i, vae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        val_stats = val(val_loader, vae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        log_stats = {'key': key_str,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
            }
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

        # Save checkpoint
        print(f"Save checkpoint {ii_key:03d}")
        torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth"))
        with (Path(params.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
        with (Path(params.output_dir) / "keys.txt").open("a") as f:
            f.write(os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth") + "\t" + key_str + "\n")
        print('\n')

def train(data_loader: Iterable, optimizer: torch.optim.Optimizer, loss_w: Callable, loss_i: Callable, vae, ldm_decoder, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        keys = key.repeat(imgs.shape[0], 1)
        
        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)
        # encode images
        
        # imgs.unsqueeze(e)
        imgs = imgs.type(torch.bfloat16)
        imgs = imgs.unsqueeze(2)
        
        imgs_z = vae.encode(imgs) # b c h w -> b z h/f w/f
        # imgs_z = imgs_z.mode()
        

        # decode latents with original and finetuned decoder
        imgs_d0 = vae.decode(imgs_z, num_frames=imgs_z.shape[2]).squeeze(2) # b z h/f w/f -> b c h w
        imgs_w = ldm_decoder.decode(imgs_z, num_frames=imgs_z.shape[2]).squeeze(2) # b z h/f w/f -> b c h w
        # pdb.set_trace()
        # extract watermark
        decoded = msg_decoder(vqgan_to_imnet(imgs_w).type(torch.float)) # b c h w -> b k
        # compute loss
        lossw = loss_w(decoded, keys)
        lossi = loss_i(imgs_w, imgs_d0)
        loss = params.lambda_w * lossw + params.lambda_i * lossi

        # optim step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log stats
        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        log_stats = {
            "iteration": ii,
            "loss": loss.item(),
            "loss_w": lossw.item(),
            "loss_i": lossi.item(),
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
            "bit_acc_avg": torch.mean(bit_accs).item(),
            "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        if ii % params.log_freq == 0:
            print(json.dumps(log_stats))
        
        # save images during training
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs.squeeze(2)),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def val(data_loader: Iterable, vae, ldm_decoder, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.eval()
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        
        imgs = imgs.to(device)
        imgs = imgs.type(torch.bfloat16)
        imgs_z = vae.encode(imgs.unsqueeze(2)) # b c h w -> b z h/f w/f
        

        imgs_d0 = vae.decode(imgs_z, num_frames=imgs_z.shape[2]).squeeze(2) # b z h/f w/f -> b c h w
        imgs_w = ldm_decoder.decode(imgs_z, num_frames=imgs_z.shape[2]).squeeze(2) # b z h/f w/f -> b c h w
        
        keys = key.repeat(imgs.shape[0], 1)

        log_stats = {
            "iteration": ii,
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
        }
        attacks = {
            'none': lambda x: x.type(torch.float),
            'crop_01': lambda x: utils_img.center_crop(x.type(torch.float), 0.1),
            'crop_05': lambda x: utils_img.center_crop(x.type(torch.float), 0.5),
            'rot_25': lambda x: utils_img.rotate(x.type(torch.float), 25),
            'rot_90': lambda x: utils_img.rotate(x.type(torch.float), 90),
            'resize_03': lambda x: utils_img.resize(x.type(torch.float), 0.3),
            'resize_07': lambda x: utils_img.resize(x.type(torch.float), 0.7),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x.type(torch.float), 1.5),
            'brightness_2': lambda x: utils_img.adjust_brightness(x.type(torch.float), 2),
            'jpeg_80': lambda x: utils_img.jpeg_compress(x.type(torch.float), 80),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x.type(torch.float), 50),
        }
        for name, attack in attacks.items():
            imgs_aug = attack(vqgan_to_imnet(imgs_w))
            decoded = msg_decoder(imgs_aug.type(torch.float)) # b c h w -> b k
            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_accs = (bit_accs == 1) # b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})

        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':

    # # generate parser / parse parameters
    # parser = get_parser()
    # params = parser.parse_args()
    
    
    params = parse_configs(training=True)

    # run experiment
    main(params)
