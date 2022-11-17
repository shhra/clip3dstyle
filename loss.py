"""Loss function used in the project.

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
import torch.nn.functional as F


def softmax3d(input):
    """Use 3D activation as described in the following paper.

    "Rethinking and Improving the Robustness of Image Style Transfer"
    Original implementation: https://github.com/peiwang062/swag/
    """
    m = torch.nn.Softmax()
    a, b, c, d = input.size()
    input = torch.reshape(input, (1, -1))
    output = m(input)
    output = torch.reshape(output, (a, b, c, d))
    return output


def argmin_cos_distance(a, b):
    """Compute the features with minimum cosine distance.

    This computes the minimum cosine distance between two feature matrices.

    Original code can be found here:
    https://github.com/Kai-46/ARF-svox2/blob/master/opt/nnfm_loss.py
    """
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)
    a_norm = ((a * a).sum(1, keepdims=True) + 1e-8).sqrt()
    a = a / (a_norm + 1e-8)
    d_mat = 1.0 - torch.matmul(a.transpose(2, 1), b)
    z_best = torch.argmin(d_mat, 2)
    return z_best


def nn_feat_replace(a, b):
    """Replace the features at "a" with best matching features from b."""
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i: i + 1], b_flat[i:i+1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n2, c, h, w)
    return z_new


def nnfm_loss(src, target):
    """Compute the nnfm loss.

    The original code from the NNFM paper used their own version of
    cosine similarity.

    More details at:
    https://github.com/Kai-46/ARF-svox2/blob/master/opt/nnfm_loss.py
    """
    loss = 0.0
    for tf, sf in zip(target, src):
        final_target = nn_feat_replace(tf, sf)
        loss += (1.0 - F.cosine_similarity(tf, final_target, dim=1)).mean()
    return loss
