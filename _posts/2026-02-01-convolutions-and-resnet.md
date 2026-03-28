---
layout: post
title: "Prehistorical DL: Convolutions and RESNET"
date: 2026-02-01
description: Some heuristics for someone completely uninsterested in CNNs
tags: CV
categories: CV
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

### Shape conventions

`(B, C, H, W)` = Batch, Channels, Height, Width

### Kernel Depth (C) Interactions (Heuristics for Cₒ Cᵢ )

The `kernel:`

- is always a square K x K
- it has the same dimension of the input channel Cᵢ

Assume you have a 3x64x64 RBG image, a single kernel will squash the image into a Hₒ, Wₒ. The output channels Cₒ is determined only by the number of kernels you have.

- Single kernel will be 3xKxK → image output is 1x64x64
- Multiple kernel, say 10, will be 10x3xKxK → image output is 10x64x64

So generelly, kernel**_s_** are Cₒ, Cᵢ, K, K

### Kernel (H,W) Ate Borders (Heuristics for H,W)

A kernel is a square K x K that eats `K-1` pixel from both H and W. This means that:

- 3x3 eats 2 pixels
  - image 64 x 64 becomes 62 x 62
- 5x5 eats 4 pixel
- 9x9 eats 8 pixel

So, what happen if we have an even-sized convolution, e.g. 6x6? The convolution will eat 3 pixel from the left or the right? from up or bottom?

- an heuristics is that it eats `"Top-Left”`

### Padding (H,W) Save Borders

Padding allow to add 0 to each side to keep borders:

- P=1 will add 1 zero to both left and right, to both up and down. So H→H+2, W→W+2
- P=2 will add 2 zero, so H→H+4, W→W+4

Using the previous info about convolution eating habits we find immediately an answer to the question: “how to keep the same size?”

- the answer is that P=(K-1)/2
  - 3x3 will need padding P=1
  - 5x5 will need padding P=2
  - 9x9 will need padding P=4

### Striding Shrinks Border (H,W)

Stride is a trick to half your H,W:

- S=1 → same H,W
- S=2 → half H,W
- S=4 → H,W are 1/4

Now a question is “Does striding shrinks is H_i, W_i / 2 or H_o, W_o / 2? that is, if i have a 64 x 64 image and P=0 and K=3, then will i get with Stride=2 31 or 32 ? “

The sad answer is that there is no exact solution, and you need to use math

Lastly, this is a good memo to imagine the stride:

> **S is computed from the left border of one step to the left border of the next step**

- **_VIZ_**
  S=1 in Conv1d
  ```python
  step 0: [a b c] d e f g h
  step 1:  a [b c d] e f g h
  step 2:  a b [c d e] f g h
  ```
  S=2 in Conv2d
  ```python
  step 0: [a b c] d e f g h
  step 1:  a b [c d e] f g h
  step 2:  a b c d [e f g] h
  ```

### The math

The formula is the following

$\text{Output} = \lfloor \frac{\text{Input} - \text{Kernel} + 2 \times \text{Padding}}{\text{Stride}} \rfloor + 1$

What you will notice is that if `stride=1` then we recover `Input-(Kernel-1)+2Padding` which is what we would expect

# RESNET

The core idea of resnet is to add skip-connections. So we have resnet block + skip connection. We just need to understand how to make the skip connections be on the same size of the output of resnet as they change quite a bit. So how can we make this well defined?

```
y = F(x) + x
```

### The ResNet Block

Simple

![convolutions-and-resnet-image.png](/assets/img/posts/convolutions-and-resnet/image.png){: width="500" height="auto" }

### Same Dimension Tricks

To make a ResNet block where the shortcut is just `x`:

- **H,W same**:
  - **FORCE IN THE BLOCK:** All convs in the block: `stride = 1`
    - For 3x3: `padding = 1` (your rule `P = (K-1)/2`)
  - **SOLVE AFTER:** via a `downsampling blocks` via stride>1 so you get
  $$
  Y=F(X)+W_sX
  $$
- **C same**:
  - **FORCE IN THE BLOCK** Design block so that the **last conv outputs the same number of channels as the input**
  - **SOLVE AFTER `by adding as many 1x1 kernel as needed`** (if K=1 eaten pixel are K-1=0)

Do that, and you get a “pure” residual block:

```
y = F(x) + x
```

with no extra layers on the shortcut.
