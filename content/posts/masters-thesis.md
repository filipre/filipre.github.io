---
title: "Master's Thesis: Asynchronous Optimization Algorithms"
date: 2020-03-13
type: "post"
tags: ["university", "math", "ai"]
weight: 2
---

Today, I finished my master's thesis with the [Computer Vision](https://vision.in.tum.de/) chair under Prof. Dr. Daniel Cremers. I surveyed several distributed, *asynchronous optimization algorithms*, gave insights into their convergence theory and performed experiments on the "delayed" versions of Stochastic Gradient Descent (SGD) and of the Alternating Direction Method of Multipliers algorithm (ADMM). You can find my thesis and its abstract below.

[![alt text](/posts/masters-thesis/thesis_cover.png)](/posts/masters-thesis/thesis.pdf)

[Download](/posts/masters-thesis/thesis.pdf)

#### Abstract

Recently, distributed optimization became more and more popular due to the increase in available training data and available computing power. There exist many distributed variants to classical optimization algorithms like Distributed Gradient Descent or Distributed ADMM but their theory often requires that computation nodes run synchronously. That means, for each iteration, all nodes must finish with their local computation and the algorithm must coordinate all nodes. This can become a bottleneck when there are slow nodes within the network, when communication breaks down or when the overhead of coordination becomes too large. *Asynchronous* optimization algorithms have gained attention within the distributed optimization community because they often overcome this bottleneck. In this thesis, we give an overview of centralized, asynchronous algorithms and perform various experiments on computer vision tasks. We look into their convergence theory but also provide examples of how we transform a given problem into a distributed one. Last but not least, we use a real computer cluster of several GPUs to optimize over a function asynchronously.
