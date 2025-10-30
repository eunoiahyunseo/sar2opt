'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys
sys.path.append(os.getcwd())
# import cv2
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

# from ram.models.ram_lora import ram
# from ram import inference_ram as inference
# from ram import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from dataloaders.naive_dataset import SAR_OPT_Dataset # 클래스 임포트

import open_clip

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)

def generate_caption(image, model, transform, accelerator):
    """Generate a caption for a given image"""
    # Load and preprocess the image
    # im = Image.open(image_path).convert("RGB")
    # im = transform(im).unsqueeze(0)
    im = image.to(accelerator.device, dtype=torch.float16)
    # print('im.dtype: ', im.dtype)
    # print('im.shape: ', im.shape)

    # Generate output using the model
    with torch.no_grad():
        generated = model.generate(im)

    # Decode and clean the generated text
    decoded_text = open_clip.decode(generated[0])
    cleaned_text = decoded_text.split("<end_of_text>")[0].replace("<start_of_text>", "").strip()
    return cleaned_text

def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.seesr_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model

def load_representation_branch(args, accelerator, dtype):

    CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name='ViT-L-14',
        pretrained='openai',
        device='cpu',
        cache_dir='cache/weights/open_clip'
    )

    checkpoint = torch.load('/home/hyunseo/workspace/kari/SAR-RS-CLIP.pt', map_location='cpu')
    CLIP_model.load_state_dict(checkpoint['state_dict']) 

    clip_visual_encoder = CLIP_model.visual
    clip_visual_encoder.to(accelerator.device)
    clip_visual_encoder.eval()
    
    vit_to_sd_proj = nn.Linear(1024, 512) # 훈련 때와 동일한 구조
    vit_to_sd_proj.load_state_dict(torch.load(os.path.join(args.seesr_model_path, "vit_to_sd_proj.pth")))
    
    clip_visual_encoder.to(accelerator.device, dtype=dtype)
    vit_to_sd_proj.to(accelerator.device, dtype=dtype)
    
    print("======== Custom ViT Model for Representation Branch Loaded ========")
    return clip_visual_encoder, vit_to_sd_proj

    
def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def main(args, enable_xformers_memory_efficient_attention=True,):
    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SeeSR")

    pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    # model = load_tag_model(args, accelerator.device)


    representation_encoder, vit_to_sd_proj = load_representation_branch(args, accelerator, dtype=torch.float16)

    validation_dataset = SAR_OPT_Dataset(
        annotations_file='/home/hyunseo/workspace/kari/SAR-TEXT/dataset/SAR-TEXT-data/annotations_QXSLAB_SAROPT.csv',
        # sar_folder='/home/hyunseo/workspace/kari/SeeSR2/datasets/train_datasets/QXSLAB_SAROPT/sar_256_oc_0.2',
        sar_folder='/home/hyunseo/workspace/kari/SAR-TEXT/dataset/SAR-TEXT-data',
        opt_folder='/home/hyunseo/workspace/kari/SeeSR2/datasets/train_datasets/QXSLAB_SAROPT/opt_256_oc_0.2',
        tokenizer=pipeline.tokenizer,
        resolution=256,
        split='val',
        null_text_ratio=0.5,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="/home/hyunseo/workspace/kari/SAR-RS-CoCa.pt",  # Path to the fine-tuned model
        weights_only=False
    )

    model.to(accelerator.device, dtype=weight_dtype)
    model.eval()

 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        for step, batch in enumerate(validation_dataloader):
            # opt_pixel_values = batch["opt_pixel_values"].to(accelerator.device, dtype=weight_dtype)
            controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
            representation_image = batch["representation_values"].to(accelerator.device, dtype=weight_dtype)
            # input_ids = batch["input_ids"].to(accelerator.device)

            ram_encoder_hidden_states = representation_encoder(representation_image)[1]
            ram_encoder_hidden_states = vit_to_sd_proj(ram_encoder_hidden_states)

            # validation_prompt = "Electro-optical Image"
            validation_prompt = generate_caption(representation_image, model, transform, accelerator)

            validation_image = controlnet_image
            negative_prompt = args.negative_prompt

            image = pipeline(
                validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, height=None, width=None,
                guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                start_point=args.start_point, ram_encoder_hidden_states=ram_encoder_hidden_states,
                latent_tiled_size=args.vae_decoder_tiled_size, latent_tiled_overlap=None,
                args=args,
            ).images[0]

            validation_image_tensor = validation_image.detach().cpu().squeeze(0)
            image_array_uint8 = (validation_image_tensor * 255)

            to_pil = transforms.ToPILImage()
            image = to_pil(image_array_uint8)

            image.save(f'{args.output_dir}/{step}.png')
            break

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seesr_model_path", type=str, default=None)
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    args = parser.parse_args()
    main(args)



