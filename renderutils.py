"""Utilites to assist in rendering.

These utilities wrap around the diffrast. The rasterization function
provides acts as closure to store the context, which in turn returns
the utility function for rendering.

Unlike test time rendering which is done from a same view point, the train
time rendering requires randomization. As a result, the `randomize` allows
to randomly change the position.

However, these randomized position should be same throughout different renders
during the style transfer. As a result, the closure stores these values.

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

import numpy as np
import nvdiffrast.torch as dr
from rasterizer import mesh, util, render


def projection(x=0.1, n=1.0, f=50.0):
    """Compute the projection matrix."""
    return np.array([[n/x,    0,            0,              0],
                     [0,   n/-x,            0,              0],
                     [0,      0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [0,      0,           -1,              0]]).astype(
                         np.float32)


def rasterization(
        eye_position=[-2.5, 0.0, -3.5],
        radius=4.0,
        power=4.5,
        batch=4,
        render_res=512,
        scale=2.0,
):
    """Generate the rasterization function for training and testing."""
    glCtx = dr.RasterizeGLContext()

    eye = np.array(eye_position)
    up = np.array([0.0, 1.0, 0.0])
    at = np.array([0, 0, 0])
    proj_mtx = projection(x=0.4, f=1000.0)

    mvp = np.zeros((batch, 4, 4),  dtype=np.float32)
    campos = np.zeros((batch, 3), dtype=np.float32)
    lightpos = np.zeros((batch, 3), dtype=np.float32)

    def randomize():
        for b in range(batch):
            # Random rotation/translation matrix for optimization.
            r_rot = util.random_rotation_translation(0.35)
            r_mv = np.matmul(util.translate(0, 0, -radius), r_rot)
            mvp[b] = np.matmul(proj_mtx, r_mv).astype(np.float32)
            campos[b] = np.linalg.inv(r_mv)[:3, 3]
            lightpos[b] = util.cosine_sample(campos[b])*radius

    def train_render(opt_mesh, materials, background=None):
        params = {'mvp': mvp,
                  'lightpos': lightpos,
                  'campos': campos,
                  'resolution': [render_res, render_res],
                  'time': 0}
        ref_mesh = mesh.compute_tangents(opt_mesh)
        ref_mesh_aabb = mesh.aabb(ref_mesh.eval())

        unit_mesh = mesh.unit_size(opt_mesh)
        train_mesh = mesh.Mesh(
            opt_mesh.v_pos, unit_mesh.t_pos_idx, material=materials,
            base=unit_mesh
        )
        train_mesh = mesh.auto_normals(train_mesh)
        train_mesh = mesh.compute_tangents(train_mesh)
        train_mesh = mesh.center_by_reference(
            train_mesh.eval(params), ref_mesh_aabb, scale)

        color_opt = render.render_mesh(glCtx,
                                       train_mesh,
                                       mvp,
                                       campos,
                                       lightpos,
                                       power,
                                       render_res,
                                       num_layers=1,
                                       background=background,
                                       min_roughness=2.0)

        return color_opt.permute(0, 3, 1, 2)

    def test_render(opt_mesh, material, background=None):
        a_mv = util.lookAt(eye, at, up)
        a_mvp = np.matmul(proj_mtx, a_mv)[None, ...]
        a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
        a_campos = np.linalg.inv(a_mv)[None, :3, 3]
        # Render the mesh at different position
        params = {'mvp': a_mvp,
                  'lightpos': a_lightpos,
                  'campos': a_campos,
                  'resolution': [render_res, render_res],
                  'time': 0}

        ref_mesh = mesh.compute_tangents(opt_mesh)
        ref_mesh_aabb = mesh.aabb(ref_mesh.eval())

        unit_mesh = mesh.unit_size(opt_mesh)
        test_mesh = mesh.Mesh(
            opt_mesh.v_pos, unit_mesh.t_pos_idx, material=material,
            base=unit_mesh
        )
        test_mesh = mesh.auto_normals(test_mesh)
        test_mesh = mesh.compute_tangents(test_mesh)
        test_mesh = mesh.center_by_reference(
            test_mesh.eval(params), ref_mesh_aabb, scale)

        color_opt = render.render_mesh(glCtx,
                                       test_mesh,
                                       a_mvp,
                                       a_campos,
                                       a_lightpos,
                                       power,
                                       render_res,
                                       num_layers=1,
                                       background=background,
                                       min_roughness=2.0)

        return color_opt.permute(0, 3, 1, 2)

    return train_render, test_render, randomize
