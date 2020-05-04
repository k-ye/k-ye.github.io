---
layout: post
title:  "Write a Performant Ray Tracer in \"Python\" (I)"
categories: [taichi, python]
mdname: "2020-04-02-write-a-performant-ray-tracer-in-python-i"
img_dir: "/static/posts/2020-04-02-write-a-performant-ray-tracer-in-python-i"
---

# Introduction

[Ray tracing](https://en.wikipedia.org/wiki/Ray_tracing_(graphics)) is a simple yet very powerful technique for doing photo-realistic rendering. It is physics based, and is one of the backbone technologies of the filming and gaming industry.

In this series of articles, we are going to write a ray tracer that renders a [Cornell Box](https://en.wikipedia.org/wiki/Cornell_box) scene (well, not an authentic one). Eventually, you should be able to get something like this on your screen:

![]({{page.img_dir}}/cornell_box.png)

# Background

I quoted Python in the title because I lied. No, we are not using Python, but a sibling language called [*Taichi*](https://github.com/taichi-dev/taichi).

> Taichi (太极) is a programming language designed for high-performance computer graphics. It is deeply embedded in Python3, and its just-in-time compiler offloads the compute-intensive tasks to multi-core CPUs and massively parallel GPUs.

In another word, Taichi offers the simplicity of Python and the performace of the hardware, the best of both.

Before you shake your head and mumble "I don't want to learn a new language just for that", don't worry. I assure you that there is nothing new about the language itself. You will be writing authentic Python, and can view Taichi just as a powerful library. In fact, here's how you'd want to install it (Note that Taichi is only available in `Python 3.6+`):

```bash
python3 -m pip install taichi
```

And [here](https://github.com/taichi-dev/taichi/blob/master/examples/fractal.py#L1-L31)'s a peak of it:

```py
import taichi as ti

ti.init(arch=ti.gpu)
n = 320
pixels = ti.var(dt=ti.f32, shape=(n * 2, n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])

@ti.kernel
def paint(t: ti.f32):
    for i, j in pixels:  # Parallized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([float(i) / n - 1, float(j) / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(n * 2, n))
for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
```

which renders a nice Julia Set:

![](https://raw.githubusercontent.com/yuanming-hu/public_files/master/graphics/taichi/fractal_small.gif)

Most of the code should look familiar (from the Python's perspective, not the algorithm it implements), except for those seemingly innocenet decorators like `@ti.func` and `@ti.kernel`. In fact, they are what Taichi is all about. There is also some boilerplate code you need to write, and we will jump into all of these in just a moment.

One more thing I should mention before we start. Here is a list of some awesome references I've found for learning ray tracers or computer graphics in general.

* [Ray Tracing in One Weekend](https://raytracing.github.io/): The content on this site is such a delight to read. It comes in three pieces, but if you only want to code something fancy, then the first brochure is more than enough to get things going. I should give 90% of the credits to the author, because most of the ray tracing works we cover here is just an unashamedly copy of the first article.
* [Phyisicall Based Rendering: From Theory To Implementation](http://www.pbr-book.org/): For anyone who is  seriously considering to work on the CG field, this book is a must read. However, it is way too advanced for the purpose of this series of posts. We will borrow some ideas from it in the third post. For now, you can safely ignore it.
* [MIT 6.837 Computer Graphics](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-837-computer-graphics-fall-2012/): This course was where I started to learn about CG. It is not only about ray tracing, but covers a broader set of topics.
* [UCSB CS 180: Introduction to Computer Graphics](https://sites.cs.ucsb.edu/~lingqi/teaching/cs180.html): I haven't taken this course yet, but the materials look more relevant and modernized compared to MIT 6.837. 该课程还有中文版，详见：[Games 101: 现代计算机图形学入门](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)
* [Kajiya, J.T., 1986, August. The rendering equation](http://www.cse.chalmers.se/edu/course/TDA362/rend_eq.pdf): The paper laying the foundation of ray tracing.
* [Veach, E. and Guibas, L.J., 1995, September. Optimally combining sampling techniques for Monte Carlo rendering](https://sites.fas.harvard.edu/~cs278/papers/veach.pdf)

Enough said, let's get started!

# Write a Basic Ray Tracer

### Hello Taichi

Before we can do anything, we need to be able to draw pixels on the screen. A screen of pixels is nothing more than a 2D array of `RGB` values, i.e. each element in the array is a tuple of three floating points. So let's create that in Taichi.

```py
import taichi as ti

sz = 800
res = (sz, sz)

ti.init(arch=ti.gpu)
color_buffer = ti.Vector(3, dt=ti.f32, shape=res)
```

If you've used `numpy` before, it may already look familiar to you of how `color_buffer` is defined. Concretely, we have instantiated a 2-D *Taichi tensor* of `shape=(800, 800)`, and assigned it to `color_buffer`. Each element in `color_buffer` is a `ti.Vector` containing three 32-bit floating point numbers (`ti.f32`).

I should also expand a bit more on `ti.init(arch=ti.gpu)`. Obviously, it does some initialization. At a minimum, you need to tell Taichi on which hardware architecture you want the Taichi program to run. There are two options here: `ti.gpu` or `ti.x64`, for targeting at GPU and CPU, respectively.

Because Taichi is designed for computer graphics (or any sort of massively parallelizable computations for that matter), it is always preferrable to target at GPU. Currently, Taichi supports these GPU platforms, and will choose the available one automatically:

* CUDA
* Apple Metal
* OpenGL, with compute shader support

However, when none of these is avaialble, Taichi will fallback to the CPU architecture. But even in that case, Taichi will still parallelize your computational tasks, just via a CPU thread pool. 

Here comes the beauty of Taichi: Once you do get access to another device with the listed GPUs support, your Taichi program will be ported to it seamlessly, and immediately enjoy the benefits of massive parallelization enabled by GPU.

Let's get back to rendering. We need to assign some color to it and display the array on the screen.

```py
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
```

Here, we have encountered our first *Taichi kernel*, `render()`. It is decorated by `@ti.kernel`. When `render()` is called for the first time, Taichi will JIT compile the decorated Python function to Taichi IR, and finally to platform specific code (LLVM for x86/64 or CUDA, Metal Shader Language, or GLSL). However, none of these details matters from the user's perspective. What you only need to remember is that, while your Taichi kernel is written in the Python syntax, internally the code is completely taken over and optimized by the Taichi compiler and runtime.

What's special about the Taichi kernel is that, during the kernel compilation, it automatically parallelizes the top-level for loop. So inside `render()`:

```py
    for u, v in color_buffer:
        color_buffer[u, v] = ti.Vector([0.678, 0.063, v / sz])
```

You can think of each `(u, v)` coordinate as being executed in its own thread [^1]. As a result, what is expected to run sequentially in Python will actually run in parallel.

This kind of for loop is called [*struct-for loops*](https://taichi.readthedocs.io/en/latest/hello.html#parallel-for-loops). It is an idiomatic way to loop over a tensor in Taichi. `(u, v)` will loop over the 2-D mesh grid as defined by the shape of `color_buffer`, i.e. `(0, 0), (0, 1), ..., (0, 799), (1, 0), ..., (799, 799)`.

Now that we have covered the essential part, the rest should be relatively easy to explain.

```py
gui = ti.GUI('Cornell Box', res)
```

This just creates a drawable window titled `Cornell Box`, with resolution `800x800`. Taichi comes with a tiny GUI system that is portable on Windows, Linux and Mac.

```py
for i in range(50000):
    render()
    img = color_buffer.to_numpy(as_vector=True)
    gui.set_image(img)
    gui.show()
```

As we can see, calling a Taichi kernel is no different from calling a plain Python function. For the first call, Taichi JIT compiles `render()` to low-level machine code, and runs it directly on GPU/CPU. Subsequent calls will skip the compilation, since it is cached by the Taichi runtime.

Once the execution completes, we need to transfer the data back. Here comes another nice thing: Taichi has already provided a few helper methods so that it can easily interact with `numpy` and `pytorch`. In our case, the Taichi tensor is converted into a `numpy` array called `img`, which is then passed into `gui` to set the color of each pixel. At last, we call `gui.show()` to update the window. Note that although `for i in range(50000):` might seem redundant, we will soon need it when doing ray tracing.

Let's name the file `box.py` and run it with `python3 box.py`. If you see the window below, congratulations! You have successfully run your first Taichi program.

![]({{page.img_dir}}/hello_taichi.png)

### Ray tracing in 5 minutes

Before more coding, we need some preliminary knowledge on how ray tracing works. If you are already an expert on this topic, feel free to jump to [the next section](#think-in-the-box). Otherwise, I'd still recommend you to read on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html). There is no way I could do a better job than that article.

First of all, to be able to describe our scene precisely in a mathematic way, we need to define a coordinate space. We will be using a *right-hand coordinate frame*. That is, your screen defines the `x` and `y` axes, where the `x` axis points to the right, and the `y` upward.
![]({{page.img_dir}}/coordinate.jpg)

### Think in the box

Now let's set up the scene. Since we are seeing inside a box, we need to define a total of five sides of the box. A side can be represented using an infinite plane. So how do we describe that mathematically?

If we are given a point on an infinite plane, $$\boldsymbol{x}$$, and its normal, $$\boldsymbol{n}$$, then for any point $$\boldsymbol{p}$$ on that plane, we know that $$\boldsymbol{p} - \boldsymbol{x}$$ must be perpendicular to $$\boldsymbol{n}$$

$$
\begin{aligned}
(\boldsymbol{p} - \boldsymbol{x}) \cdot \boldsymbol{n} = \boldsymbol{0}  
\end{aligned}
$$

Since $$\boldsymbol{p}$$ is also along the ray, i.e. $$\boldsymbol{p}(t) = \boldsymbol{o} + \boldsymbol{d}\cdot t$$, we can have:

$$
\begin{aligned}
(\boldsymbol{p}(t) - \boldsymbol{x}) \cdot \boldsymbol{n} & = \boldsymbol{0} \\
(\boldsymbol{o} + \boldsymbol{d} \cdot t - \boldsymbol{x}) \cdot \boldsymbol{n} & = \boldsymbol{0} \\
(\boldsymbol{d} \cdot \boldsymbol{n}) \cdot t & = (\boldsymbol{x} - \boldsymbol{o}) \cdot \boldsymbol{n} \\
t & = \frac{(\boldsymbol{x} - \boldsymbol{o}) \cdot \boldsymbol{n}}{\boldsymbol{d} \cdot \boldsymbol{n}}
\end{aligned}
$$

We can now define a function to check if a ray would intersect with a given plane:

```py
@ti.func
def ray_plane_intersect(pos, d, pt_on_plane, norm):
    t = inf
    denom = ti.dot(d, norm)
    if abs(denom) > eps:
        t = ti.dot((pt_on_plane - pos), norm) / denom
    return t
```

Here, we have seen a new Taichi decorator, `@ti.func`. This one is handled in a similar way as `@ti.kernel` is, in that the decorated function is compiled to native machine code, and executed without the Python interpreter. Because of this, functions decorated with `@ti.func` can be called from another function that is decorated by either `@ti.kernel` or `@ti.func`. However, you should not invoke this function inside the Python scope.

Most of the procedure inside `ray_plane_intersect()` is a direct translation of the equations above. Note that there is a possibility where the ray direction $$\boldsymbol{d}$$ is in parallel with the plane. In another word, it is perpendicular to the plane normal $$\boldsymbol{n}$$. So we compute $$t$$ only if that is not the case, i.e. $$\boldsymbol{d} \cdot \boldsymbol{n} \ge \epsilon$$.

In the previous section, we have explained that when shooting a ray, we need to find out the closest point it hits. Let's define such a function:

```py
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

    return closest, normal, color, mat
```

`closest` and `normal` are used to store the final $$t$$ and the normal of the hit surface, respectively. `color` and `mat` store the color and the material of the hit surface. If nothing is hit, these value will not be modified. One restriction that I should point out: Currently, Taichi function doesn't support multi-return yet. So you will have to pre-define all the variables that will be returned at the top. A bit cumbersome, but not a big deal.

As a start, we have only defined the left side plane of the box. This plane passes point `[-1.1, 0.0, 0.0]` and has a normal of `[1.0, 0.0, 0.0]`. In another word, this is a $$yz$$ plane pointing to the $$+x$$ direction, and intersecting the $$x$$ axis at `-1.1`. Its material is lambertian (`mat_lambertian`), with a red color (`RGB = [0.65, 0.05, 0.05]`).

TODO: Add a graph showing yz plane pointing to +x

We can add the right, top, bottom and far side planes in a very similar way. Below is the complete function definition:

```py
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
```

We summarize the properties of each side in the following table.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
@media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;}}</style>
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-0pky">side</th>
    <th class="tg-0pky">axis-aligned plane</th>
    <th class="tg-0pky">normal</th>
    <th class="tg-0pky">intersecting axis</th>
    <th class="tg-0pky">color</th>
  </tr>
  <tr>
    <td class="tg-0pky">left</td>
    <td class="tg-0pky">yz</td>
    <td class="tg-0pky">+x</td>
    <td class="tg-0pky">x = -1.1</td>
    <td class="tg-0pky">[0.65, 0.05, 0.05] (Red)</td>
  </tr>
  <tr>
    <td class="tg-0pky">right</td>
    <td class="tg-0pky">yz</td>
    <td class="tg-0pky">-x</td>
    <td class="tg-0pky">x = 1.1</td>
    <td class="tg-0pky">[0.12, 0.45, 0.15] (Green)</td>
  </tr>
  <tr>
    <td class="tg-0pky">bottom</td>
    <td class="tg-0pky">xz</td>
    <td class="tg-0pky">+y</td>
    <td class="tg-0pky">y = 0.0</td>
    <td class="tg-0pky">[0.93, 0.93, 0.93] (Gray)</td>
  </tr>
  <tr>
    <td class="tg-0pky">top</td>
    <td class="tg-0pky">xz</td>
    <td class="tg-0pky">-y</td>
    <td class="tg-0pky">y = 2.0</td>
    <td class="tg-0pky">[0.93, 0.93, 0.93] (Gray)</td>
  </tr>
  <tr>
    <td class="tg-0pky">far</td>
    <td class="tg-0pky">xy</td>
    <td class="tg-0pky">+z</td>
    <td class="tg-0pky">z = 0.0</td>
    <td class="tg-0pky">[0.93, 0.93, 0.93] (Gray)</td>
  </tr>
</table></div>

### Oh shoot

We have done a great amount of work so far, yet the main `render()` kernel remains unchanged. If you run `box.py` now, it still renders that mundane gradient color view.

*"Oh shoot! How am I supposed to see my hard work?"*

Exactly, let's shoot some rays into the scene :)

We need some helpers from `math` and `numpy`, let's import them first.

```py
# import taichi as ti
import numpy as np
import math
```

We also need to define a few constants, some of which will be explained later.

```py
# ...
# res = (sz, sz)
aspect_ratio = res[0] / res[1]

eps = 1e-4
inf = 1e10
fov = 0.8
max_ray_depth = 10

# ...
# mat_light = 2

camera_pos = ti.Vector([0.0, 0.6, 3.0])
light_color = ti.Vector([0.9, 0.85, 0.7])

# ti.init(arch=ti.gpu)
```

With these in place, we are ready to bulk up `render()`.

```py
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
```

The first ray always starts at the chosen camera position. The ray direction is more interesting. Let's take a look at the y component first:

```py
(2 * fov * (v + ti.random()) / res[1] - fov)
```

Ignoring the randomness (`ti.random()`) for now, `v` is in range `[0, res[1])`, hence the range of this expression is `[-fov, fov)`.

For the x component, we see that `fov` is multiplied with `aspect_ratio = res[0] / res[1]`. Because `u` is in `[0, res[0])`, the entire expression's range is `[-fov * aspect_ratio, fov * aspect_ratio)`. `aspect_ratio` makes the viewport adapt to the GUI resolution automatically. This isn't particular useful in our example, since we have created a square window. However, you can try a heterogenous resolution, e.g. `(1000, 800)`, and verify that the the scene covers a wider angle horizontally.

The randomness is introduced for *antialiasing*. Every time a pixel is sampled, we jitter the first ray's direction a bit, so that it can hit at a slightly different location. The pixel's final color is then the average of all the samples, and gradually converges as the sampling iteration continues. This is a very common approach in ray tracing for removing some artifacts (e.g. jagged edges, as shown in [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html#antialiasing)).

We need a few variables to hold the states of the ray tracing process.

```py
        # ...
        # ray_dir = ti.normalized(ray_dir)

        px_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])
        depth = 0
```

Let's revisit the ray tracing mechanism: Until we have reached the `max_ray_depth`, we shoot one ray from the previous hitting point (or the camera position, if this is the first tray). If the ray hits nothing, then we can terminate the loop earlier. If it hits a light source, we will still jump out of the loop. But before that, we multiply . Otherwise, according to our settings, it means the ray has hit one of the box surfaces, which is a lambertian material. In this situation, we just randomly pick a new ray direction for the next iteration.

```py
        # ...
        # depth = 0

        while depth < max_ray_depth:
            closest, hit_normal, hit_color, mat = intersect_scene(ray_pos, ray_dir)
            if mat == mat_none:
                break
            if mat == mat_light:
                px_color = throughput * light_color
                break
            hit_pos = ray_pos + closest * ray_dir
            depth += 1
            ray_dir = sample_ray_dir(ray_dir, hit_normal, hit_pos)
            ray_pos = hit_pos + eps * ray_dir
            throughput *= hit_color

        color_buffer[u, v] += px_color
```

Most of the code should be self explanatory by now, we still need to address a few details here.

1. If `mat == mat_none`, `px_color` will not be set. From a physics point of view, the entire path we have traced didn't get a chance to hit any light source, hence the pixel is not lit.
2. If `mat == mat_light`, this means the traced path has hit a light source. `px_color` will be attenuated by all the materials that the ray has hit along its path, i.e. `throughput * light_color`.
3. The new ray direction is computed by `sample_ray_dir`, which we haven't given its definition yet.
4. Note how we compoute the new ray origin, `ray_pos`. We have shifted the hit position by a tiny offset along the new direction, i.e. `eps * ray_dir`. Without this shift, because of the numerical precision errors, sometimes the new ray would immediately intersect with the surface it has just hit, the so called *self-intersection*. This could result in very noisy artifacts where the objects are covered with small black dots. [Here](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/ligth-and-shadows) is an example to illustrate that artifacts.

Let's get back to `sample_ray_dir`. We've briefly explained that a new ray direction should be picked randomly over the unit hemisphere above the hit surface. (Note that this claim is correct *only for lambertian materials*. In the later posts we will handle specular and transmissive materials, where the new ray direction cannot be chosen arbitrarily.) One strategy for doing this is to choose a random point within the sphere that is centered at $$\boldsymbol{p} + \boldsymbol{n}$$, where $$\boldsymbol{p}$$ is the hit point and $$\boldsymbol{n}$$ is the unit surface normal.

```py
@ti.func
def random_in_unit_sphere():
    p = ti.Vector([0.0, 0.0, 0.0])
    while True:
        for i in ti.static(range(3)):
            p[i] = ti.random() * 2 - 1.0
        if p.dot(p) <= 1.0:
            break
    return p
```

What this function does is to keep proposing a 3-D point inside a `[-1.0, +1.0]` cube randomly, and terminates when the distance between the point and the zero origin is less than or equal to `1.0`.

You may be wondering what `ti.static` is all about. Well, it tells Taichi to unroll your loop at compile time. This is a neat trick to gain performance without code duplication.

With this helper, we can easily define how to sample a new direction:

```py
@ti.func
def sample_ray_dir(indir, normal, hit_pos):
    return ti.normalized(random_in_unit_sphere() + normal)
```

In its most canonical form, this should have been:

```py
hit_pos + ti.normalized(random_in_unit_sphere() + normal) - hit_pos
```

But then `hit_pos` gets canceled out. Albeit none of the args except for `normal` is used, we still pass them in because they will be in later posts.

Here's the complete snippet for the ray tracing procedure:

```py
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
```

Let's give it a shot, `python3 box.py`:

![]({{page.img_dir}}/all_dark.png)

*"Whaaaa?"*

Recall how the scene is set up so far. We have created five planes,  all of which have a lambertian material. However, no light source has ever been defined! All the rays have bounced happily for a while, then gone in vein.

Let's quickly hack up a light source. I'm going to switch the material of the top plane from `mat_lambertian` to `mat_light`, a one line change.

```py
@ti.func
def intersect_scene(ray_o, ray_d):
    # ...

    # top
    # ...
    # if 0 < cur_dist < closest:
        # ...
        color, mat = gray, mat_light  # mat_lambertian
```

If we run the example again:

![]({{page.img_dir}}/final.png)

*Voila*, our hard working has finally paid off!

I'm going to end the post here, because this should be enough for one with no CG experience to digest for a while. If you have followed this far, thank you for your time and patience. In the next post, I will be adding the rest of the components to the scene, including a *bounded* rectangular light source, and two spheres with apparently different materials. If you are yearning for more, think about how to check for ray-rectangle and/or ray-sphere intersections, and try extending the example for yourself :)

---


[^1]: This is only true on GPU, since on CPU, the number of threads we can launch is usually much less than the size of the tensor. On the other hand, CPU cores are much more powerful and can finish the computation per coordinate much faster.