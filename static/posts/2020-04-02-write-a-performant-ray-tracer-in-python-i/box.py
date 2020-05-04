import taichi as ti

sz = 800
res = (sz, sz)

mat_none = 0
mat_lambertian = 1
mat_light = 2

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


@ti.kernel
def render():
    for u, v in color_buffer:
        color_buffer[u, v] = ti.Vector([0.678, 0.063, v / sz])


gui = ti.GUI('Cornell Box', res)
for i in range(50000):
    render()
    img = color_buffer.to_numpy(as_vector=True)
    gui.set_image(img)
    gui.show()

input("Press any key to quit")
