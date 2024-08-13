import argparse
import json
import os
from glob import glob
import pdb

from mmengine.config import Config


def parse_args(training=False):
    parser = argparse.ArgumentParser()
    
    
    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, help="Path to the training data directory", required=False)
    aa("--val_dir", type=str, help="Path to the validation data directory", required=False)

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="sd/stable-diffusion-v-1-4-original/v1-inference.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_ckpt", type=str, default="sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt", help="Path to the checkpoint file for the LDM model") 
    aa("--msg_decoder_path", type=str, default= "models/hidden/dec_48b_whit.torchscript.pt", help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=0.2, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--output_save_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")

    
    

    # model config
    parser.add_argument("--config", help="model config file path")

    # ======================================================
    # General
    # ======================================================
    parser.add_argument("--seed", default=None, type=int, help="seed for reproducibility")
    parser.add_argument(
        "--ckpt-path",
        default=None,
        type=str,
        help="path to model ckpt; will overwrite cfg.model.from_pretrained if specified",
    )
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")
    parser.add_argument("--outputs", default=None, type=str, help="the dir to save model weights")
    parser.add_argument("--flash-attn", default=None, type=str2bool, help="enable flash attention")
    parser.add_argument("--layernorm-kernel", default=None, type=str2bool, help="enable layernorm kernel")
    parser.add_argument("--resolution", default=None, type=str, help="multi resolution")
    parser.add_argument("--data-path", default=None, type=str, help="path to data csv")
    parser.add_argument("--dtype", default=None, type=str, help="data type")
    parser.add_argument("--watermarked_eval", default=False, action="store_true", help="watermarked eval")
    parser.add_argument("--watermarked_vae_ckpt", default=None, type=str, help="watermarked_vae_ckpt")

    # ======================================================
    # Inference
    # ======================================================
    if not training:
        # output
        parser.add_argument("--save-dir", default=None, type=str, help="path to save generated samples")
        parser.add_argument("--sample-name", default=None, type=str, help="sample name, default is sample_idx")
        parser.add_argument("--start-index", default=None, type=int, help="start index for sample name")
        parser.add_argument("--end-index", default=None, type=int, help="end index for sample name")
        parser.add_argument("--num-sample", default=None, type=int, help="number of samples to generate for one prompt")
        parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")
        parser.add_argument("--verbose", default=None, type=int, help="verbose level")

        # prompt
        parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
        parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")
        parser.add_argument("--llm-refine", default=None, type=str2bool, help="enable LLM refine")
        parser.add_argument("--prompt-generator", default=None, type=str, help="prompt generator")

        # image/video
        parser.add_argument("--num-frames", default=None, type=str, help="number of frames")
        parser.add_argument("--fps", default=None, type=int, help="fps")
        parser.add_argument("--save-fps", default=None, type=int, help="save fps")
        parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")
        parser.add_argument("--frame-interval", default=None, type=int, help="frame interval")
        parser.add_argument("--aspect-ratio", default=None, type=str, help="aspect ratio (h:w)")
        parser.add_argument("--watermark", default=None, type=str2bool, help="watermark video")

        # hyperparameters
        parser.add_argument("--num-sampling-steps", default=None, type=int, help="sampling steps")
        parser.add_argument("--cfg-scale", default=None, type=float, help="balance between cond & uncond")

        # reference
        parser.add_argument("--loop", default=None, type=int, help="loop")
        parser.add_argument("--condition-frame-length", default=None, type=int, help="condition frame length")
        parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
        parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
        parser.add_argument("--aes", default=None, type=float, help="aesthetic score")
        parser.add_argument("--flow", default=None, type=float, help="flow score")
        parser.add_argument("--camera-motion", default=None, type=str, help="camera motion")
    # ======================================================
    # Training
    # ======================================================
    else:
        parser.add_argument("--lr", default=None, type=float, help="learning rate")
        parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
        parser.add_argument("--load", default=None, type=str, help="path to continue training")
        parser.add_argument("--start-from-scratch", action="store_true", help="start training from scratch")
        parser.add_argument("--warmup-steps", default=None, type=int, help="warmup steps")
        parser.add_argument("--record-time", default=False, action="store_true", help="record time of each part")

    return parser.parse_args()


def merge_args(cfg, args, training=False):
    if args.ckpt_path is not None:
        cfg.model["from_pretrained"] = args.ckpt_path
        if cfg.get("discriminator") is not None:
            cfg.discriminator["from_pretrained"] = args.ckpt_path
        args.ckpt_path = None
    if args.flash_attn is not None:
        cfg.model["enable_flash_attn"] = args.flash_attn
        args.enable_flash_attn = None
    if args.layernorm_kernel is not None:
        cfg.model["enable_layernorm_kernel"] = args.layernorm_kernel
        args.enable_layernorm_kernel = None
    if args.data_path is not None:
        cfg.dataset["data_path"] = args.data_path
        args.data_path = None
    # NOTE: for vae inference (reconstruction)
    if not training and "dataset" in cfg:
        if args.image_size is not None:
            cfg.dataset["image_size"] = args.image_size
        if args.num_frames is not None:
            cfg.dataset["num_frames"] = args.num_frames
    if not training:
        if args.cfg_scale is not None:
            cfg.scheduler["cfg_scale"] = args.cfg_scale
            args.cfg_scale = None
        if args.num_sampling_steps is not None:
            cfg.scheduler["num_sampling_steps"] = args.num_sampling_steps
            args.num_sampling_steps = None

    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    return cfg


def read_config(config_path):
    cfg = Config.fromfile(config_path)
    return cfg

def parse_configs_latte(training=False):
    return parse_args(training)


def parse_configs(training=False):
    args = parse_args(training)
    cfg = read_config(args.config)
    cfg = merge_args(cfg, args, training)
    return cfg


def define_experiment_workspace(cfg, get_last_workspace=False):
    """
    This function creates a folder for experiment tracking.

    Args:
        args: The parsed arguments.

    Returns:
        exp_dir: The path to the experiment folder.
    """
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(cfg.outputs, exist_ok=True)
    experiment_index = len(glob(f"{cfg.outputs}/*"))
    if get_last_workspace:
        experiment_index -= 1

    # Create an experiment folder
    model_name = cfg.model["type"].replace("/", "-")
    exp_name = f"{experiment_index:03d}-{model_name}"
    exp_dir = f"{cfg.outputs}/{exp_name}"
    return exp_name, exp_dir


def save_training_config(cfg, experiment_dir):
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
