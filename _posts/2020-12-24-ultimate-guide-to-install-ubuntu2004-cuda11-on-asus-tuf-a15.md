---
layout: post
title:  "Ultimate Guide to Install Ubuntu 20.04 LTS and CUDA 11 on ASUS TUF A15"
categories: [cuda]
---

This post summarizes my sweat and tears when trying to install Ubuntu and CUDA on the ASUS TUF A15 laptop. I started with a simple intention of just getting CUDA up and running ASAP, without any particular distribution version in mind. This soon turned out to be naive and over-optimistic. I've learnt my lessons, and hopefully nobody would suffer from this, ever...

---

Here's the concrete spec of the laptop:

* Laptop model: ASUS Gaming Laptop TUF Gaming A15 FA506IV
* CPU: AMD Ryzen 9 4900H
* GPU: NVIDIA GeForce RTX 2060 (discrete), with an integrated AMD GPU that I didn't care but was able to cause great agony.
* WiFi: Realtek Semiconductor Co., Ltd. - RTL8822CE 802.11ac PCIe Wireless Network Adapter

These are what I have installed in the end:

* Ubuntu: `20.04.1 LTS`
* Kernel version: `5.8.0-23-generic`
* CUDA version: `11.1.0`
* NVIDIA graphics driver: `nvidia-driver-455`

## Why these versions?

I started with Ubuntu 18.04 LTS, feeling that this would be a safe bet due to its wider adoption rate. I've made a mistake already, as the canonical ISO did not come with the WiFi driver. It wasn't a hard blocker, since you can always fall back to the ethernet. I then proceeded to follow the CUDA installation guide. As of 12/2020, the latest CUDA release is [`11.2.0`](https://developer.nvidia.com/cuda-downloads). Here came the second problem:

1. At first, I followed [NVIDIA's offical guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal) and then `sudo apt-get -y install cuda`. According to [the guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-metas):

> The `cuda` meta pacakage installs all CUDA Toolkit and Driver packages. Handles upgrading to the next version of the cuda package when it's released.

So it took care of both the CUDA SDK and the lower-level NVIDIA graphics drivers, nice!?

Unfortunately, the NVIDIA graphics driver `460` shipped in this release didn't seem to play well with the kernel. When I followed this, I always ended up with the boot screen stuck at `boot error ucsi_acpi USBC000:00: PPM init failed (-110)`.

2. Installing NVIDIA graphics driver first, then `sudo apt-get install nvidia-cuda-toolkit`.

This might have worked if all your goal was to just get the NVCC compiler. Unfortunately, it drifted too far from the offical CUDA setup, and the rest of my toolchain couldn't get along.
    
In addition, note that it is [*not* recommended](https://forums.developer.nvidia.com/t/cuda-10-installation-problems-on-ubuntu-18-04/68615/2) to mix the official's CUDA installation with your pre-installed NVIDIA graphics drivers. [This post](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux) has listed a few popular ways to install NVIDIA graphics drivers:

1. `ubuntu-drivers devices`
1. From a non-official PPA repo `ppa:graphics-drivers/ppa`
1. From an official runfile.

IIUC, any NVIDIA graphics driver you have installed via 1 or 2 will have to be uninstalled before the offical CUDA guide can work. On top of that, NVIDIA also [discourages](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#choose-installation-method) the usage of runfile installation.

So the problem space became a bit clear:

* The Linux kernel I chose was too ancient to even offer me WiFi. (Yeah, I know that updating the Linux firmware is an option.)
* The CUDA I chose was too new, but we should still follow the official installation guide, as opposed to installing from the packages maintained by Ubuntu.

# Steps

1. Download the latest Ubuntu distribution from https://ubuntu.com/download/desktop. I used Ubuntu `20.04.1 LTS`, which thankfully came with the WiFi driver.

    After installing the OS, it asks you to reboot. Note that at this point, the OS has no NVIDIA graphics driver at all. So whenever a reboot is required, we must change the kernel launch paramter.

   1. At the GRUB screen, press `e`.
   1. At the line starting with `linux /boot/...`, replace `quiet splash` with `nomodeset`.
   1. If none of the above words makes sense, here's a much better answer with images: https://askubuntu.com/a/38834.

1. The default ISO had a kernel version `5.4.0-58-generic`. This one doesn't have enough support for the integrated AMD GPU, which meant that my external monitor connected via HDMI wouldn't work at all. The solution was to just upgrade the kernel. Here I picked `5.8.0-23-generic` (after trying `5.6`-something...):

```bash
sudo apt get linux-headers-5.8.0-23-generic linux-image-5.8.0-23-generic linux-modules-5.8.0-23-generic linux-modules-extra-5.8.0-23-generic 
```

Reboot again. Don't forget to set `nomodeset`.

I also tried upgrading the AMD GPU drivers following [its offcial guide](https://amdgpu-install.readthedocs.io/en/latest/index.html), but never got it working.

1. `sudo apt install build-essential`: This gives you `gcc` and `make`.
1. Install CUDA!

For those brave souls that want to continue drilling down, you have my deepest respect and full blessing...