"""Main optimization script.

Copyright 2022, Shailesh Mishra

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

from logs import OptimizationLogger
from pathlib import Path
from loss import nnfm_loss
from renderutils import rasterization
from features import FeatureExtractor, VGGFeatures
from utils import load_image, load_mesh, extract_mean_colors, save_image
from rasterizer import texture, material
from copy import deepcopy
import imageio
import torch.nn.functional as F
import argparse
import clip
import torch
import time
import numpy as np
import random
import os
import warnings

warnings.filterwarnings("ignore")

random.seed(143)
torch.manual_seed(2143)
np.random.seed(11423)


# There is a weird bug with the differentiable render that doesn't render
# the mesh, when scoping is enabled i.e. Wrapping the optimization into some
# `run()` function and calling the `run()` function from main.
#
# To get over the bug, the code is being implemented as a script that runs the
# final optimization.

parser = argparse.ArgumentParser()
parser.add_argument(
    "--style-image", type=str, help="Path to the input image.", required=True
)

parser.add_argument(
    "--multi-image",
    type=str,
    nargs="+",
    help="Path to the input image.",
    default=[],
    required=False,
)

parser.add_argument(
    "--input-mesh", type=str, help="Path to the input mesh.", required=True
)

parser.add_argument(
    "--texture-res",
    type=int,
    help="Texture resolution of the final mesh texture.",
    default=1024,
    required=False,
)

parser.add_argument(
    "--eye-position",
    type=float,
    nargs="+",
    help="Center of the camera.",
    default=[2.5, 0.0, -3.5],
    required=False,
)

parser.add_argument(
    "--radius",
    type=float,
    help="Camera radius from the origin.",
    default=4.0,
    required=False,
)

parser.add_argument(
    "--power",
    type=float,
    help="The power of light being used in the scene.",
    default=3.0,
    required=False,
)

parser.add_argument(
    "--scale",
    type=float,
    help="The power of light being used in the scene.",
    default=3.0,
    required=False,
)

parser.add_argument(
    "--batch-size",
    type=int,
    help="Number of images being rendered by rasterizer.",
    default=4,
    required=False,
)

parser.add_argument(
    "--render-res",
    type=int,
    help="Resolution of rendered image.",
    default=512,
    required=False,
)

parser.add_argument(
    "--output-path", type=str, help="Output path for the given scene.", required=True
)

parser.add_argument(
    "--log-iteration",
    type=int,
    help="Log the temporary result during optimization.",
    default=500,
    required=False,
)

parser.add_argument(
    "--iteration",
    type=int,
    help="Number of steps to optimize the image for.",
    default=15000,
    required=False,
)

parser.add_argument("--use-vgg", action="store_true")

parser.add_argument(
    "--index", type=str, help="Index for mutex", default="done", required=False
)

args = parser.parse_args()

# Store the gpu rank and local rank for identifying main GPU.
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
if world > 1:
    print("Setting up the world.")
    torch.distributed.init_process_group("NCCL")
    print("World is ready.")
torch.cuda.set_device(local_rank)


# Download the imageio plugins only on rank zero instance.
if rank == 0:
    print(args.input_mesh, args.style_image)
    imageio.plugins.freeimage.download()


# Check outputpath
if rank == 0 and not Path(args.output_path).exists():
    os.makedirs(args.output_path)

device = torch.device(local_rank)
clip_model, _ = clip.load("RN50", device)

# Load the style image
style_image = load_image(args.style_image, args.render_res).unsqueeze(0).cuda()

data_mesh = load_mesh(args.input_mesh)

logger = OptimizationLogger(args.output_path)

# Extract the style features from the style image
style_layers = [
    "layer3_0_conv2",
    "layer3_1_conv2",
    "layer3_2_conv2",
    "layer3_4_conv2",
    "layer3_5_conv2",
    "layer4_0_conv2",
    "layer4_1_conv2",
    "layer4_2_conv2",
]

# Depending on the flag, use clip based features or vgg based features
if not args.use_vgg:
    if rank == 0:
        logger.info("Using CLIP based features")
    feature_extractor = FeatureExtractor(
        clip_model.visual.requires_grad_(False), style_layers
    ).cuda()
    style_weight = 5e4
    content_weight = 40.0
else:
    if rank == 0:
        logger.info("Using VGG based features")
    feature_extractor = VGGFeatures(device)
    style_weight = 200.0
    content_weight = 1.0

vgg_extractor = VGGFeatures(device)

if len(args.multi_image) > 0:
    # Consume the data here to merge the style features
    image = load_image(args.multi_image[0], args.render_res).unsqueeze(0).cuda()
    features = feature_extractor(image)
    for x in args.multi_image[1:]:
        image = load_image(x, args.render_res).unsqueeze(0).cuda()
        new_features = feature_extractor(image)
        for i in range(len(features)):
            features[i] = torch.cat((features[i], new_features[i]), axis=2)
    style_features = features
else:
    with torch.no_grad():
        style_features = feature_extractor(style_image)

for each in style_features:
    each = each.repeat(args.batch_size, 1, 1, 1)

# Generate a color palette from the original source image
mean_colors = extract_mean_colors(style_image[0]).cuda()

# create gradient background
gradient_bg = mean_colors[0].clone().cpu().numpy()
gradient_bg = gradient_bg.reshape(1, -1, 3)
gradient_bg = np.vstack([gradient_bg] * int(args.render_res))
gradient_bg = np.hstack([gradient_bg] * int(args.render_res / gradient_bg.shape[1]))
gradient_bg = (
    torch.from_numpy(gradient_bg).float().cuda().unsqueeze(0).permute(0, 2, 3, 1)
)

# Intialize the background
background = style_image.clone().detach().permute(0, 2, 3, 1)

# Intialize the material
texture_res = [args.texture_res, args.texture_res]
kd_map = texture.create_trainable(deepcopy(data_mesh.material["kd"]), texture_res, True)
ks_map = texture.create_trainable(deepcopy(data_mesh.material["ks"]), texture_res, True)
train_material = material.Material(
    {
        "bsdf": data_mesh.material["bsdf"],
        "kd": kd_map,
        "ks": ks_map,
    }
)


# Initialize for multi gpu training.
class Trainer(torch.nn.Module):
    """A wrapper around nn.Module to store the optimization parameters.

    Pytorch requries trainable parameters for distrubuted training. Using
    nn.Module, we are able to control the trainable parameters.
    """

    def __init__(self, train_material):
        """Initialize the trainer."""
        super(Trainer, self).__init__()
        self.kd = texture.create_trainable(
            train_material["kd"].getMips()[0], texture_res, True
        )
        self.ks = texture.create_trainable(
            train_material["ks"].getMips()[0], texture_res, True
        )
        self.material = material.Material(
            {
                "bsdf": data_mesh.material["bsdf"],
                "kd": self.kd,
                "ks": self.ks,
            }
        ).requires_grad_(True)
        self.params = self.material.parameters()

    def forward(
        self,
        train_render,
        style_features,
    ):
        """Perform loss computation on the forward loop.

        This ensures that are gradients are being passed.
        """
        with torch.no_grad():
            content_image = train_render(data_mesh, data_mesh.material, None)
            # visualize(content_image, i)
        if len(args.multi_image) > 0:
            style_image_path = args.multi_image[
                random.randint(0, len(args.multi_image) - 1)
            ]
            style_image_in = (
                load_image(style_image_path, args.render_res).unsqueeze(0).cuda()
            )
            background = style_image_in.clone().detach().permute(0, 2, 3, 1)
        else:
            background = style_image.clone().detach().permute(0, 2, 3, 1)

        # Extract the scene features
        target_image = train_render(data_mesh, self.material, background)

        # Compute the loss style loss using ResNet Feature extractor
        target_style_features = feature_extractor(target_image)

        style_loss = style_weight * nnfm_loss(style_features, target_style_features)

        # Compute the loss content loss using VGG feature extractor
        src_content_features = vgg_extractor(content_image)
        target_content_features = vgg_extractor(target_image)

        content_loss = 0.0
        for (tc, sc) in zip(target_content_features, src_content_features):
            content_loss = content_loss + content_weight * F.mse_loss(tc, sc)

        # Compute the color loss
        color_image = self.material.kd.getMips()[0].permute(0, 2, 3, 1)
        color_image = color_image.view(-1, 3)

        weight = (0.083 * num_iterations) * torch.exp(
            torch.tensor(max(0.5, 2 - (i * 2) / num_iterations)).cuda()
        )
        color_losses = torch.cdist(color_image, mean_colors).min(dim=1)[0]
        mean_loss = weight * color_losses.mean()
        # var_loss = weight * color_losses.var()
        color_loss = mean_loss
        total_loss = style_loss + content_loss + color_loss
        return total_loss, style_loss, content_loss, color_loss


trainer_noddp = Trainer(train_material).cuda()

if world > 1:
    trainer = torch.nn.parallel.DistributedDataParallel(
        trainer_noddp, device_ids=[device]
    )
    rank = 0
else:
    trainer = trainer_noddp

# Intialize the optimizer and schedulers
optimizer = torch.optim.Adam(trainer.parameters(), lr=0.0095)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

# Get the render functions
train_render, test_render, randomize = rasterization(
    eye_position=args.eye_position,
    radius=args.radius,
    power=args.power,
    batch=args.batch_size,
    render_res=args.render_res,
    scale=args.scale,
)


num_iterations = args.iteration
print(f"Starting for {num_iterations}!!!")
if world > 1:
    torch.distributed.barrier()
print("All gpus are synced.")
# Start the optimization loop
for i in range(num_iterations + 1):
    randomize()
    optimizer.zero_grad()
    start = time.time()
    # Render a scene
    total_loss, style_loss, content_loss, color_loss = trainer(
        train_render, style_features
    )
    # Backpropagate the loss
    total_loss.backward()
    # for name, params in trainer.named_parameters():
    #     logger.info("Checking grad. {} {}".format(name, params.grad))

    optimizer.step()
    scheduler.step()

    if rank == 0 and i % args.log_iteration == 0:
        delta = time.time() - start

        logger.info(
            "{}/{}".format(
                time.strftime("%H:%M:%S", time.gmtime(delta * i)),
                time.strftime("%H:%M:%S", time.gmtime(delta * num_iterations)),
            )
        )
        log_info = (
            "At iteration {}, total loss = {}, style_loss = {},"
            "content_loss = {} and color_loss = {}."
        )
        logger.info(
            log_info.format(i, total_loss, style_loss, content_loss, color_loss)
        )
        if world > 1:
            updated = trainer.module.material
        else:
            updated = trainer.material
        output_image = test_render(
            data_mesh, updated, torch.ones_like(background).cuda()
        )
        save_image(output_image, args.output_path, i)
        # output_image = train_render(data_mesh, updated,
        #                             torch.ones_like(background).cuda())
        # save_image(output_image, args.output_path, i + 1)
        # updated = trainer.module.material
        texture = updated["kd"].getMips()[0][0]
        imageio.imwrite(
            Path(args.output_path).joinpath("trained_texture.exr"),
            texture.detach().cpu().numpy(),
        )

if world > 1:
    torch.distributed.barrier()

# Let there be this file
if rank == 0:
    Path(args.index).touch()
    print("Optimization done for index ", args.index)
    if world > 1:
        torch.distributed.destroy_process_group()
    raise SystemExit
