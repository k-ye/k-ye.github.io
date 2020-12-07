---
layout: post
title:  "MeshMore: a (delisted) iOS App for Low-Poly Arts"
categories: [ios, low-poly]
res_dir: "/static/posts/2020-12-06-meshmore-an-ios-app-for-low-poly-arts"
---

# The Movitation

Back in mid-2017, I was obsessed with *low-poly arts*. I eargly searched for ways to paint in this artstyle, concluded that there were two major approaches, and experimented a bit with each of them.

The first approach is quite straightforward. You simply need to load an image into Photoshop, draw lots of triangles on top of that image and fill each triangle with the color averaged from the covered region. [This video](https://www.youtube.com/watch?v=oQf6ivOgoMs) can get you started in 10 minutes. Repeatitive? Sure. But the end result can be quite amazing.

![]({{page.res_dir}}/ps-horse.png)

The second one has a somewhat higher entry barrier: you need to bring everything to the table on your own. That is, sculpturing the 3D meshes, placing them in the viewport, setting up the lights, and repeating the whole process until you are satisfied (or exhausted). Being an engineer with probably less than ten aesthetic brain cells, this seemingly is not something I can excel at. However, Blender has made this a lot easier than it sounds. In the end, I managed to pull off these works following [this video tutorial](https://www.youtube.com/watch?v=JjW6r10Mlqs).

![]({{page.res_dir}}/rhino.jpg)
![]({{page.res_dir}}/got-drogon.png)

# The App

Fast forwarding to 2018 Q4, I once again felt the urge to create low-poly arts. So this time, I wrote an iOS app implementing the first approach, which allowed me and my friends to draw on our phones. The name of the app was *MeshMore*, in the hope that people would both draw and learn more about meshes, a simple yet omnipresent data structure. Here are some outcomes (with post-processing filtering applied):

![]({{page.res_dir}}/reindeer.jpg)
![]({{page.res_dir}}/iceage-diego.jpg)
![]({{page.res_dir}}/ny-sol.jpg)
![]({{page.res_dir}}/horse-mbp.jpg)

The core data structure I used was called the *half-edge mesh*. It is a very efficient data structure to support operations like travesing the adjacent elements in the mesh, edge splitting and collapsing, etc. Some of the materials that I found useful during the development are listed below.

* [https://462cmu.github.io/asst2_meshedit](https://462cmu.github.io/asst2_meshedit)
* [https://kaba.hilvi.org/homepage/blog/halfedge/halfedge.htm](https://kaba.hilvi.org/homepage/blog/halfedge/halfedge.htm): Unfortunately the website doesn't seem to be updated for a while. So if you click on it, your browser is probably gonna give you a big warning. But the material helped clarify a lot of the concepts and implementation details.

Hopefully this short post has provided something fun.

![]({{page.res_dir}}/zootopia-sloth.jpg)