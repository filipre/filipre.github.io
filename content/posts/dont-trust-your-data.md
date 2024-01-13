---
title: "Do Not Blindly Trust Your Data"
date: 2018-08-01
# description: "A blog post"
type: "post"
tags: ["university"]
---

During my masters, I took the class "Data Science Society" which is about the ethics on data science. For the final project of this class we had to write an essay on a related topic. This was my attempt.

[![alt text](/posts/dont-trust-your-data/essay_cover.png)](/posts/dont-trust-your-data/essay.pdf)

[Download](/posts/dont-trust-your-data/essay.pdf)

<!--

*Note: This essay was part of the "Data Science Society" class that deals with ethical questions in data science. At the end, each student had to choose a related topic and write a short essay about it. This was my attempt.*


In Data Science and Machine Learning one does not simply apply an algorithm and reads out the result to solve a problem. Often, it includes tasks like data collection, data cleaning and data interpretation. In all these steps, many things can go wrong and manipulate the final result. Because people trust more and more the results of an algorithm, it is important to understand what can go wrong, why a result might be biased and how such a wrong result affects other people. In this essay we will analyze some factors and argue about their importance and influence. At the end we reason why data science is more than only applying mathematical operations.

## Introduction

Today, the role of data science and machine learning becomes more and more important because it gives us either new insights to existing problems or it enables us to write programs that sometimes even outperform human beings. In the industry there is a high demand of data scientists that can effectively explain given data sets and solve certain tasks like classification, prediction and generation. Machine Learning already proved its strength in industries like pharmacy and health [[1](do-not-blindly-trust-data#references)]. Usually, data is extracted from some source, cleaned from errors or noise and then fed into a fitting algorithm producing a result which needs to be interpreted at the end. However, many things can go wrong if one is not careful enough. For any scientist or engineer it is important to understand the challenges in each step of the pipeline in order not to give wrong answers or even end up in dangerous solutions.

We can ask ourselves: What are the things we need to be aware of in order to make our analysis as accurate as possible? Why is there sometimes a bias in our data sets and what can we do against it? And what are the consequences of our (wrong) results on the society? In this essay we are going to answer these question by taking a closer look at algorithms and their unexpected behavior. Then, we observe an bias if experiments are conducted in public or anonymously. Last but not least we raise awareness of being critical towards publicized papers.

## Dangers of Trusting an Algorithm blindly

The term "algorithm" can refer to many different things depending on which person one talks to [[2](do-not-blindly-trust-data#references)]. A computer scientist might refer to the definition of an algorithm using a Turing machine while a journalist might refer to the behavior of certain software. Nevertheless, in both cases one needs to be aware of what the algorithm exactly does and what not.

### Machine Learning Algorithms do not Always Solve the Task

An algorithm solves a given problem by performing predefined steps in a certain order. The result of an algorithm is a solution to the given problem. There are many types of algorithms. Deterministic algorithms like the sorting algorithm "Bubble Sort" will *always* provide the same output for a given input. In contrast, a probabilistic algorithm like the "Fast Johnson Lindenstrauss Transformation" returns a solution that only holds with a high probability. Machine Learning (and statistical) algorithms like the "Linear Regression" describe another type of algorithms which differ even more: Let us assume we want to predict the housing price of a flat in Munich if one gives us the square meter size of it. The algorithm itself does not give us an answer to that but instead returns a mathematical model which needs to be asked instead. In order to do so, the algorithm needs "training data" to train the model that we want to return. Otherwise, the model does not know anything about Munich or even about housing prices in general and would return random-like estimates instead.

This is a problem. The first two categories of algorithms we mentioned above did not rely on training data because they were solving the problem explicitly. However, machine learning algorithms generate a solution by looking at existing solutions and then try to make one on their own, implicitly. Suddenly, the performance of a learning algorithm depends on the training data.

Solving a problem implicitly is not a bad thing per se. Actually, it is often much more easier to solve a problem that way than describing a solution explicitly. But, it adds a danger that one can easily overlook: The algorithm always produces a model (even for bad or wrong training data) and just because it returns does not mean it also returns the right solution! While for "classical" algorithms there often exist proofs guaranteeing correctness, users of machine learning algorithms must be aware that these do not exist in the same way. The user must not have a false sense of security just because the algorithm returned successfully and the underlying challenge is how to evaluate a model.

There are ways to perform the evaluation and we will look at them shortly. But the consequence of this is quite depressing: In the real world there might be bad machine learning models that do not act the way which was intended. For example, in \cite{dailymail_com_2017} a soap dispenser was trained on detecting hands in order to dispose soap. It turned out that the model was trained on a data set that only contained white colored hands and at the end it could not detect the hand gesture by a black person. If we translate this problems where human life is in danger, the magnitude of it becomes clear. Imagine a self-driving car that cannot detect a group of humans because the model expected a different physical appearance. These kind of issues are not obvious from the start and might surprise.

TOOD

## References

[1] Machine Learning Healthcare Applications - 2018 and Beyond - Faggella, Daniel - 2018

[2] Digital keywords: a vocabulary of information society and culture - Peters, Benjamin - Princeton University Press. - 2016

-->
