import taichi as ti

sz = 800
res = (sz, sz)

ti.init(arch=ti.gpu)
color_buffer = ti.Vector(3, dt=ti.f32, shape=res)


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
