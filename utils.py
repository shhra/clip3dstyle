"""Basic utilities function to handle saving and loading of the data.

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

import torch
import imageio
import torchvision
import torchvision.transforms as T

from pathlib import Path
from kmeans_pytorch import kmeans
from rasterizer import obj
from PIL import Image


def load_image(path, size=512):
    """Load image from give path."""
    image = Image.open(path)
    image = T.Compose([
        T.Resize((size, size)),
        T.ToTensor()
    ])(image)
    return image[0:3, :, :]


def load_mesh(path):
    """Create a mesh loader."""
    ext = Path(path).suffix
    if ext == ".obj":
        return obj.load_obj(str(path), clear_ks=True, mtl_override=None)
    assert False, "Invalid mesh file extension"


def extract_mean_colors(image):
    """Compute major colors using K-Means."""
    x = image.clone().detach().view(3, -1).transpose(0, 1)
    uniques = torch.unique(x, dim=0)
    _, cluster_centers = kmeans(
        X=uniques, num_clusters=16, distance='euclidean', device=torch.device('cuda')
    )
    return cluster_centers


def save_test(test_render, data_mesh, train_material, background, output_path, i=0):
    """Render a new image at given for ith iteration."""
    image = test_render(data_mesh, train_material, background)
    torchvision.utils.save_image(image, str(
        Path(output_path).joinpath(f"test_{i}.png")))


def save_image(image, output_path, i=0):
    """Save the image at given index i."""
    torchvision.utils.save_image(image, str(
        Path(output_path).joinpath(f"image_{i}.png")))


def save_texture(output_path, train_material):
    """Save the texture at desired location."""
    imageio.plugins.freeimage.download()
    texture = train_material["kd"].getMips()[0][0]
    final_path = Path(output_path).joinpath("trained_texture.exr")
    imageio.imwrite(str(final_path), texture.detach().cpu().numpy())
