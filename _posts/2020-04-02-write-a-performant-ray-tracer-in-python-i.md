---
layout: post
title:  "Write a Performant Ray Tracer in \"Python\" (I)"
categories: [taichi, python]
mdname: "2020-04-02-write-a-performant-ray-tracer-in-python-i"
img_dir: "/static/posts/2020-04-02-write-a-performant-ray-tracer-in-python-i"
---

# Introduction

In this series of articles, we are going to write a ray tracer that renders a [Cornell Box](https://en.wikipedia.org/wiki/Cornell_box) scene (well, not an authentic one). Eventually, you should be able to get something like this on your screen:

![]({{page.img_dir}}/cornell_box.png)

# Background

I quoted Python in the title because I lied. No, we are not using Python, but a sibling language called [*Taichi*](https://github.com/taichi-dev/taichi).

> Taichi (太极) is a programming language designed for high-performance computer graphics. It is deeply embedded in Python3, and its just-in-time compiler offloads the compute-intensive tasks to multi-core CPUs and massively parallel GPUs.

In another word, Taichi offers the simplicity of Python and the performace of the hardware, the best of both.

Before you shake your head and mumble "I don't want to learn a new language just for that", don't worry. I assure you that there is nothing new about the language itself. You will be writing native Python, and can view Taichi just as a powerful library. In fact, here's how you'd want to install it:

```bash
pip install taichi
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

* [Ray Tracing in One Weekend](https://raytracing.github.io/): The content on this site is such a delight to read. It comes in three pieces, but if you only want to code something fancy, then the first brochure is more than enough to get things going. I should give 95% of the credits to the author, because most of the ray tracing works we cover is just an unashamedly copy of the first part of the trilogy.
* [Phyisicall Based Rendering: From Theory To Implementation](http://www.pbr-book.org/): For anyone who is  seriously considering to work on the CG field, this book is a must read. However, it is way too advanced for the purpose of this series of posts. We will borrow some ideas from it in the third post. For now, you can safely ignore it.
* [MIT 6.837 Computer Graphics](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-837-computer-graphics-fall-2012/): This course was where I started to learn about CG.
* [UCSB CS 180: Introduction to Computer Graphics](https://sites.cs.ucsb.edu/~lingqi/teaching/cs180.html): I haven't taken this course yet, but the materials look more relevant and modernized compared to MIT 6.837. 该课程还有中文版，见：[Games 101: 现代计算机图形学入门](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html).
* [Kajiya, J.T., 1986, August. The rendering equation](http://www.cse.chalmers.se/edu/course/TDA362/rend_eq.pdf): The paper laying the foundation of ray tracing.
* [Veach, E. and Guibas, L.J., 1995, September. Optimally combining sampling techniques for Monte Carlo rendering](https://sites.fas.harvard.edu/~cs278/papers/veach.pdf)

Enough said, let's get started!

# Write a Basic Ray Tracer

### Hello Taichi

Before we can do anything, we need to be able to draw pixels on the screen. A screen of pixels is nothing more than a 2D array of `RGB` values, i.e. each element in the array is a tuple of three floating points. So let's create that in Taichi.

```py
import taichi as ti

ti.init(arch=ti.gpu)
sz = 800
res = (sz, sz)
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

As we can see, calling a Taichi kernel (`render()`) is no different from calling a plain Python function. For the first call, Taichi JIT compiles `render()` to low-level machine code, and runs it directly on GPU/CPU. Subsequent calls will skip the compilation, since it is cached by the Taichi runtime.

Once the execution completes, we need to transfer the data back. Here comes another nice thing: Taichi has already provided a few helper methods so that it can easily interact with `numpy` and `pytorch`. In our case, the Taichi tensor is converted into a `numpy` array called `img`, which is then passed into `gui` to set the color of each pixel. At last, we call `gui.show()` to update the window. Note that although `for i in range(50000):` might seem redundant, we will soon need it when doing ray tracing.

Let's name the file `box.py` and run it with `python3 box.py`. If you see the window below, congratulations! You have successfully run your first Taichi program.

![]({{page.img_dir}}/hello_taichi.png)

### Ray tracing in 5 minutes

Before more coding, we need some preliminary knowledge on how ray tracing works. If you are already an expert on this topic, feel free to jump to [the next section](#think-in-the-box). Otherwise, I'd still recommend you to read on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html). There is no way I could do a better job than that article.

![]({{page.img_dir}}/coordinate.jpg)

### Think in the box

Now let's set up the scene. First of all, we are seeing inside a box
### Oh shoot

[^1]: This is only true on GPU, since on CPU, the number of threads we can launch is usually much less than the size of the tensor. On the other hand, CPU cores are much more powerful and can finish the computation per coordinate much faster.