import taichi as ti
import numpy as np
import math

sz = 800
res = (800, 800)
aspect_ratio = res[0] / res[1]

eps = 1e-4
inf = 1e10
fov = 0.8
max_ray_depth = 10

mat_none = 0
mat_lambertian = 1
mat_light = 2

camera_pos = ti.Vector([0.0, 0.6, 3.0])
light_color = ti.Vector([0.9, 0.85, 0.7])

ti.init(arch=ti.gpu)
color_buffer = ti.Vector(3, dt=ti.f32, shape=res)


@ti.func
def ray_plane_intersect(ray_o, ray_d, pt_on_plane, norm):
    t = inf
    denom = ti.dot(ray_d, norm)
    if abs(denom) > eps:
        t = ti.dot((pt_on_plane - ray_o), norm) / denom
    return t


@ti.func
def intersect_scene(ray_o, ray_d):
    closest, normal = inf, ti.Vector.zero(ti.f32, 3)
    color, mat = ti.Vector.zero(ti.f32, 3), mat_none

    # left
    pnorm = ti.Vector([1.0, 0.0, 0.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([-1.1, 0.0, 0.0]),
                                   pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = ti.Vector([0.65, 0.05, 0.05]), mat_lambertian
    # right
    pnorm = ti.Vector([-1.0, 0.0, 0.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([1.1, 0.0, 0.0]),
                                   pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = ti.Vector([0.12, 0.45, 0.15]), mat_lambertian
    # bottom
    gray = ti.Vector([0.93, 0.93, 0.93])
    pnorm = ti.Vector([0.0, 1.0, 0.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([0.0, 0.0, 0.0]),
                                   pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = gray, mat_lambertian
    # top
    pnorm = ti.Vector([0.0, -1.0, 0.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([0.0, 2.0, 0.0]),
                                   pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = gray, mat_lambertian
    # far
    pnorm = ti.Vector([0.0, 0.0, 1.0])
    cur_dist = ray_plane_intersect(ray_o, ray_d, ti.Vector([0.0, 0.0, 0.0]),
                                   pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        color, mat = gray, mat_lambertian

    return closest, normal, color, mat


@ti.func
def random_in_unit_sphere():
    p = ti.Vector([0.0, 0.0, 0.0])
    while True:
        for i in ti.static(range(3)):
            p[i] = ti.random() * 2 - 1.0
        if p.dot(p) <= 1.0:
            break
    return p


@ti.func
def sample_ray_dir(indir, normal, hit_pos, mat):
    return ti.normalized(random_in_unit_sphere() + normal)


@ti.kernel
def render():
    for u, v in color_buffer:
        ray_pos = camera_pos
        ray_dir = ti.Vector([
            (2 * fov * (u + ti.random()) / res[1] - fov * aspect_ratio),
            (2 * fov * (v + ti.random()) / res[1] - fov),
            -1.0,
        ])
        ray_dir = ti.normalized(ray_dir)

        px_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])
        depth = 0

        while depth < max_ray_depth:
            closest, hit_normal, hit_color, mat = intersect_scene(ray_pos, ray_dir)
            if mat == mat_none:
                break
            if mat == mat_light:
                px_color = throughput * light_color
                break
            hit_pos = ray_pos + closest * ray_dir
            depth += 1
            ray_dir = sample_ray_dir(ray_dir, hit_normal, hit_pos, mat)
            ray_pos = hit_pos + eps * ray_dir
            throughput *= hit_color

        color_buffer[u, v] += px_color


gui = ti.GUI('Cornell Box', res)
for i in range(5000000):
    render()
    img = color_buffer.to_numpy(as_vector=True) * (1 / (i + 1))
    img = np.sqrt(img / img.mean() * 0.24)
    gui.set_image(img)
    gui.show()

input("Press any key to quit")
