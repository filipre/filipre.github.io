---
layout: post
title: "Sum of Manhattan Distances in Ordered Array"
date: 2020-05-14 15:00:00 +0100
---

When we look at an algorithm and compare it with other ones, we are almost always interested in how long it will take to solve a given problem. Instead of measuring the physical time from start to finish, we measure the amount of operations they perform. Using "number of operations" instead of "time" has the advantage that we don't need to specify or fix certain computer hardware and we can reason about the algorithm's performance without even running it. Because counting the operations can become tricky for more complicated algorithms, we are also not intersted in the *exact* number of operations but only in a rough approximation. For example, if someone gives us a list of 100 numbers and wants us to sort them, one method could use about 10000 operations while the other method may only use as much as 700. Clearly, we would prefer the second method. If you are a mathematician or computer scientist, then you already know that we use the [Big $$\mathcal{O}$$ notation](https://en.wikipedia.org/wiki/Big_O_notation) to denote and approximate the growth of something . However, do you also recall why 


the related other notations and their definitions? I didn't. This post also helps me to remember the difference between Small $$\mathcal{o}$$ and Big $$\mathcal{O}$$. 


and also some fundamentals in Analysis, like the difference between $$\max$$ and $$\sup$$ or the difference between $$\lim$$ and $$\limsup$$/$$\liminf$$. Then, in the second part of this post, I will take an algorithm interview question (one you may get when you apply for a programmer job somehwere) and solve it using a slow algorithm and a fast one.

## Analysis 101

Let's start with some basics in analysis. 

sequence

converge or diverge

max/min and sup/inf


TODO Introduction / Speed


## Big O Notation

There is a very general definition of the Big $$\mathcal{O}$$ notation which works for any [topological space](https://en.wikipedia.org/wiki/Topological_space) but in the context of algorithms, we can use a much simpler definition.



## Example

TODO: leetcode submission [https://leetcode.com/contribute/20137](https://leetcode.com/contribute/20137)

TODO: introduction, leetcode

### Problem

### Brute Force Solution

$$\mathcal{O}(n^2)$$ TODO.

```python
def bruteforce_distances(self, arr: List[int]) -> List[int]:
    distances = [0]*len(arr)
    for i, a in enumerate(arr):
        distance = 0
        for a_ in arr:
            distance += abs(a - a_)
        distances[i] = distance
    return distances
```

### Efficient Solution

$$\mathcal{O}(n)$$ TODO

TODO

```
Brute Force O(n^2) - not accepted:
There are n elements in the array. 
For each element (outer loop), calculate the sum of all Manhattan distances to all (other) elements by using an inner loop.
Due to the nested loop, we end up with a runtime of O(n^2).

Efficient Solution O(n) - accepted:
We make several observations: First, the array is sorted. Second, we have that 
    |a[k]-[i]| = |a[k]-a[j]| + |a[j]-a[i]| for 0 <= i < j < k < n
Let's imagine the problem as a graph problem, where each element is represented by a node 
and each distance between two elements is the weight of an edge between two nodes. We end up with a complete graph.
The sum of Manhattan distances is simply the sum of all weights of the edges for a particular node.
Due to the second property above, we know that a lot of information is redundantly given. 
For example, we can imply the weight of edge 0-2 by summing the weights of edge 0-1 and 1-2.
In fact, we only need the distances between two consecutive elements a[i] and a[i+1] for 0 <= i < n-1.
Let d(i,j) represent the Manhattan distance between two nodes
For the first node, we see following pattern:
D[0] = 0 + d(0,1) + d(0,2) + ... + d(0,n) = 0 + d(0,1) + (d(0,1) + d(1,2)) + (d(0,1) + d(1,2) + d(2,3)) + ... + (d(0,1) + ... + d(n-1, n))
For the second node, we notice that:
D[1] = d(0,1) + 0 + d(1,2) + ... + d(1,n) = d(0,1) + 0 + d(1,2) + (d(1,2) + d(2,3)) + ... + (d(1,2) + ... + d(n-1, n))
For the last node we find out:
D[n-1] = d(0,n) + d(1,n) + ... + d(n-1, n) + 0 = (d(0,1) + ... + d(n-1, n)) + (d(1,2) + ... + d(n-1, n)) + ... + d(n-1, n)
For increasing elements, we see that the term on the right side of the "0" decreases and the term on the left side of the "0" increases.
By carefully counting the amount of d(i,i+1) terms and using two cumulative sums "forward" and "backward", we can reconstruct any D[i].
```

TODO graph example

TODO detailed formula in $$\KaTeX$$.

----

$$
\begin{aligned}
D_0 &= 0 + w_{0,1} + w_{0,2} + \dotsb + w_{0,n} \\
     &= 0 + (w_{0,1}) + (w_{0,1} + w_{1,2}) + \dotsb + (w_{0,1} + w_{1,2} + \dotsb + w_{n-1,n}) \\
     &= 0 + n \cdot w_{0,1} + (n-1) \cdot w_{1,2} + \dotsb + 1 \cdot w_{n-1,n} \\
\;\\
D_1 &= w_{0,1} + 0 + w_{1,2} + w_{1,3} + \dotsb + w_{1,n} \\
     &= (w_{0,1}) + 0 + (w_{1,2}) + (w_{1,2} + w_{2,3}) + \dotsb + (w_{1,2} + \dotsb + w_{n-1,n}) \\
     &= 1 \cdot w_{0,1} + 0 + (n-1) \cdot w_{1,2} + (n-2) \cdot w_{2,3} + \dotsb + 1 \cdot w_{n-1,n} \\
\;\\
D_2 &= w_{0,2} + w_{1,2} + 0 + w_{2,3} + w_{2,4} + \dotsb + w_{2,n} \\
&= (w_{0,1} + w_{1,2}) + (w_{1,2}) + 0 + (w_{2,3}) + (w_{2,3} + w_{3,4}) + \dotsb + (w_{2,3} + \dotsb + w_{n-1,n}) \\
&= 1 \cdot w_{0,1} + 2 \cdot w_{1,2} + 0 + (n-2) \cdot w_{2,3} + \dotsb + 1 \cdot w_{n-1,n}
\end{aligned}
$$







```python
def distances(self, arr: List[int]) -> List[int]:
    n = len(arr)
    if n < 2:
        return [0]*n
    relative_distances = [ arr[i]-arr[i-1] for i in range(1, n) ]
    forward, backward = [0]*n, [0]*n
    for i in range(1, n):
        forward[i] = forward[i-1] + i*relative_distances[i-1]
        backward[n-i-1] = backward[n-i] + i*relative_distances[n-i-1]
    sum_distances = [0]*n
    for i in range(n):
        sum_distances[i] = forward[i] + backward[i]
    return sum_distances
```


### Unordered Array

TODO: Investigate


### Full Code

```python
class Solution:
    def bruteforce_distances(self, arr: List[int]) -> List[int]:
        distances = [0]*len(arr)
        for i, a in enumerate(arr):
            distance = 0
            for a_ in arr:
                distance += abs(a - a_)
            distances[i] = distance
        return distances

    # arr is sorted
    def distances(self, arr: List[int]) -> List[int]:
        n = len(arr)
        if n < 2:
            return [0]*n
        relative_distances = [ arr[i]-arr[i-1] for i in range(1, n) ]
        forward, backward = [0]*n, [0]*n
        for i in range(1, n):
            forward[i] = forward[i-1] + i*relative_distances[i-1]
            backward[n-i-1] = backward[n-i] + i*relative_distances[n-i-1]
        sum_distances = [0]*n
        for i in range(n):
            sum_distances[i] = forward[i] + backward[i]
        return sum_distances

if __name__ == "__main__":
    solution = Solution()
    testcases = [
        [0,0,0,0,0,0,0,0,0,0,0],
        [1,2,3,4,5,6,7,8,9,10],
        [1,4,7,12,64,67,123],
        [1,1,2,2,3,3,4,4,5,5,6,6,7,7,7],
        [0,1,2,4,8,16,32,64,128,256],
        [100],
        [],
        [100,200],
        [100,200,300]
    ]

    for testcase in testcases:
        print("Testcase: ", testcase)
        print("bruteforce", solution.bruteforce_distances(testcase))
        print("efficient ", solution.distances(testcase))
        print()
```

Output

```
Testcase:   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Bruteforce: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Efficient:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Testcase:   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Bruteforce: [45, 37, 31, 27, 25, 25, 27, 31, 37, 45]
Efficient:  [45, 37, 31, 27, 25, 25, 27, 31, 37, 45]

Testcase:   [1, 4, 7, 12, 64, 67, 123]
Bruteforce: [271, 256, 247, 242, 294, 303, 583]
Efficient:  [271, 256, 247, 242, 294, 303, 583]

Testcase:   [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7]
Bruteforce: [48, 48, 37, 37, 30, 30, 27, 27, 28, 28, 33, 33, 42, 42, 42]
Efficient:  [48, 48, 37, 37, 30, 30, 27, 27, 28, 28, 33, 33, 42, 42, 42]

Testcase:   [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
Bruteforce: [511, 503, 497, 489, 481, 481, 513, 641, 1025, 2049]
Efficient:  [511, 503, 497, 489, 481, 481, 513, 641, 1025, 2049]

Testcase:   [100]
Bruteforce: [0]
Efficient:  [0]

Testcase:   []
Bruteforce: []
Efficient:  []

Testcase:   [100, 200]
Bruteforce: [100, 100]
Efficient:  [100, 100]

Testcase:   [100, 200, 300]
Bruteforce: [300, 200, 300]
Efficient:  [300, 200, 300]
```