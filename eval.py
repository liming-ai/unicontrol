'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
'''

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from dataset_eval import MyDataset
from cldm.model import create_model, load_state_dict
from pathlib import Path
import jsonlines
import argparse
import pdb
from PIL import Image
import numpy as np
import einops
import os
from cldm.ddim_unicontrol_hacked import DDIMSampler
import random
from torchvision.utils import make_grid
from utils import check_safety, get_reward_model
from datasets import load_dataset
import torchvision.transforms.functional as F
import cv2

parser = argparse.ArgumentParser(description="args")
parser.add_argument("--task", type=str, default='canny', choices=['canny', 'hed', 'seg', 'normal', 'depth','openpose', 'imageedit', 'bbox', 'hedsketch', 'outpainting', 'grayscale', 'blur', 'inpainting', 'grayscale'], help='option of task')
parser.add_argument("--ckpt", type=str, default='./ckpts/unicontrol.ckpt', help='$path to checkpoint')
parser.add_argument("--dataset_name", type=str, default='limingcv/Captioned_ADE20K')
parser.add_argument("--cache_dir", type=str, default='data/huggingface_datasets')
parser.add_argument("--split", type=str, default='validation')
parser.add_argument("--prompt_column", type=str, default='text')
parser.add_argument("--condition_column", type=str, default='control_depth')
parser.add_argument("--strength", type=float, default=1.0, help='control guidiance strength')
parser.add_argument("--scale", type=float, default=9.0, help='text guidiance scale')
parser.add_argument("--output_path", type=str, default='./output', help='$path to save prediction results')
parser.add_argument("--config", type=str, default='./models/cldm_v15_unicontrol.yaml', help='option of config') 
parser.add_argument("--guess_mode", default=False, help='Guess Mode') 
parser.add_argument("--seed", default=-1, help='Random Seed') 
parser.add_argument("--save_memory", default=False, help='Low Memory') 
parser.add_argument("--num_samples", type=int, default=4, help='Num of Samples') 
parser.add_argument("--n_prompt", type=str, default='worst quality, low quality', help='negative prompts') 
parser.add_argument("--ddim_steps", default=50, help='DDIM Steps') 

args = parser.parse_args()

# Configs
checkpoint_path = args.ckpt
batch_size = 1
seed = args.seed
num_samples = args.num_samples
guess_mode = args.guess_mode
n_prompt = args.n_prompt
ddim_steps=args.ddim_steps

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(args.config).cpu()
model.load_state_dict(load_state_dict(checkpoint_path, location='cpu'), strict=False) #, strict=False

task=args.task


output_dir = os.path.join(args.output_path, 'scale'+str(int(args.scale)), task)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

control_key = 'control_' + task

path_meta= "data/"
# task_name = task if task != 'seg' else 'segbase'
task_name = task
path_json = "data/" + task_name + ".json"

target_list = []
with jsonlines.open(Path( path_json)) as reader:
    for ll in reader:
        target_list.append(ll[control_key].split('/')[1])

print(f"Length of target list is {len(target_list)}")

model.eval()

dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir, split=args.split)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
ddim_sampler = DDIMSampler(model)

sample_path = os.path.join(output_dir, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))
grid_count = len(os.listdir(output_dir)) - 1

task_to_instruction = {
    "hed": "hed edge to image",
    "canny": "canny edge to image",
    "seg": "segmentation map to image",
    "depth": "depth map to image",
    "openpose": "human pose skeleton to image"
}

a_prompt = 'best quality, extremely detailed'
# Inference loop
with torch.no_grad():
    for idx, data in enumerate(dataset):

        prompt = data[args.prompt_column]

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if args.save_memory:
            model.low_vram_shift(is_diffusing=False)

        if args.task == 'canny':
            low_threshold = np.random.randint(0, 255)
            high_threshold = np.random.randint(low_threshold, 255)

            control = data[args.condition_column].convert('RGB').resize((512, 512))
            control = cv2.Canny(np.array(control), low_threshold, high_threshold)
            control = F.pil_to_tensor(Image.fromarray(control).convert('RGB'))
            control = control.squeeze(0).cuda() / 255.0
        elif args.task == 'hed':
            annotator = get_reward_model(
                task="hed",
                model_path="https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
            )
            annotator.eval()
            image = data['image'].convert("RGB").resize((512, 512))
            image = F.pil_to_tensor(image).unsqueeze(0) / 255.0
            with torch.no_grad():
                control = annotator(image).squeeze(0).cuda()
            control = control.repeat(3, 1, 1)
        else:
            control = data[args.condition_column].convert('RGB')
            control = F.pil_to_tensor(control.resize((512, 512))) / 255.0
            control = control.squeeze(0).cuda()

        if len(control.shape) == 2:
            control = control.unsqueeze(0)

        C, H, W = control.shape
        control = torch.stack([control for _ in range(num_samples)], dim=0)

        task_dic = {}
        task_dic['name'] = f'control_{args.task}'
        task_instruction = task_to_instruction[args.task]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        un_cond = {"c_concat": [torch.zeros_like(control)] if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)]}
        shape = (4, H // 8, W // 8)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=0,
                                                     unconditional_guidance_scale=args.scale,
                                                     unconditional_conditioning=un_cond)
        x_samples = model.decode_first_stage(samples)

        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

        for local_id in range(num_samples):
            if not os.path.exists(os.path.join(sample_path, f'group_{local_id}')):
                os.makedirs(os.path.join(sample_path, f'group_{local_id}'))

        x_checked_image, has_nsfw_concept = check_safety(x_samples)
        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
        for group_id, x_sample in enumerate(x_checked_image_torch):
            x_sample = 255. * einops.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))

            if args.task == 'canny':
                generated_canny = cv2.Canny(np.array(img), low_threshold, high_threshold)
                generated_canny = Image.fromarray(generated_canny)
                generated_canny.save(os.path.join(sample_path, f'group_{group_id}', f"{idx}_canny.png"))

            if args.task == 'hed':
                generated_hed = F.pil_to_tensor(img).unsqueeze(0) / 255.0
                with torch.no_grad():
                    generated_hed = annotator(generated_hed).squeeze(0).cuda()

                generated_hed = F.to_pil_image(generated_hed)
                generated_hed.save(os.path.join(sample_path, f'group_{group_id}', f"{idx}_hed.png"))

            img.save(os.path.join(sample_path, f'group_{group_id}', f"{idx}.png"))
            print(os.path.join(sample_path, f'group_{group_id}', f"{idx}.png"))
            base_count += 1
        control_img = F.to_pil_image(control[0])
        control_img.save(os.path.join(sample_path, f"{idx}_{args.task}.png"))
        print(os.path.join(sample_path, f"{idx}_{args.task}.png"))


# python3 eval.py --task='seg' --dataset_name='limingcv/Captioned_ADE20K' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='prompt' --condition_column='control_seg'

# python3 eval.py --task='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='text' --condition_column='control_depth'

# python3 eval.py --task='canny' --dataset_name='limingcv/MultiGen-20M_canny_eval' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='text' --condition_column='image'

# python3 eval.py --task='hed' --dataset_name='limingcv/MultiGen-20M_canny_eval' --cache_dir='data/huggingface_datasets' --split='validation' --prompt_column='text' --condition_column='image'
