---
layout: post
title: "Expected Distance between two Points"
date: 2018-01-24 18:50:00 +0900
---

## Square

Monte carlo

TODO: simple Expectation rules... expectation of uniform distribution , variance, ...

$$
\mathbb{E}[X] = \frac{a+b}{2}
$$

$$
\mathbb{V}[X] = \frac{(b-a)^2}{12}
$$

$$
\mathbb{V}[X] = \mathbb{E}[X^2] - E[X]^2 \Leftrightarrow
\mathbb{E}[X^2] = \mathbb{V}[X] + E[X]^2 = \frac{1}{12} + \frac{1}{4} = \frac{1}{3}
$$

Easy:

$$
\begin{aligned}
   \mathbb{E}[d^2]
   &= \mathbb{E}[(X_2-X_1)^2+(Y_2-Y_1)^2] \\
   &= \mathbb{E}[(X_2-X_1)^2] + \mathbb{E}[(Y_2-Y_1)^2] \\
   &= \mathbb{E}[X_2^2 - 2 X_2 X_1 + X_1^2] + \mathbb{E}[Y_2^2 -2 Y_2 Y_1 + Y_1^2] \\
   &= \mathbb{E}[X_2^2] - 2 \mathbb{E}[X_2 X_1] + \mathbb{E}[X_1^2] + \mathbb{E}[Y_2^2] - 2 \mathbb{E}[Y_2 Y_1] +  \mathbb{E}[Y_1^2] \\
   &= \mathbb{E}[X_2^2] - 2 \mathbb{E}[X_2] \mathbb{E}[X_1] + \mathbb{E}[X_1^2] + \mathbb{E}[Y_2^2] - 2 \mathbb{E}[Y_2] \mathbb{E}[Y_1] + \mathbb{E}[Y_1^2] \\
   &= \frac{1}{3} - 2 \cdot \frac{1}{2} \cdot \frac{1}{2} + \frac{1}{3} + \frac{1}{3} - 2 \cdot \frac{1}{2} \cdot \frac{1}{2} + \frac{1}{3} \\
   &= \frac{1}{3}
\end{aligned}
$$

### $$n$$-dimensional

Hard:

TODO

## Circle
