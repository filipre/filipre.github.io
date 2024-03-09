---
title: 'Calculating Prime Numbers in SQL'
date: 2024-01-28T21:51:39+01:00
description: "Implementing the Sieve of Eratosthenes in SQL using Recursive Expressions"
type: "post"
tags: ["sql", "math", "interview", "algorithms"]
weight: 2
---

Ever needed to calculate prime numbers in a SQL database? No? Here is the query anyways:

```sql
with recursive primes(i, d) as (
  select generate_series(2, 10000), 2
  union
  select i, (select min(i) from primes where i > d)
  from primes
  where d <= (select sqrt(max(i)) from primes) and (i <= d or mod(i, d) != 0)
)

select i
from primes
where d = (select max(d) from primes)
order by primes.i ASC
```

You can try it out [here](https://hyper-db.de/interface.html). To confirm it, there are [1229](https://jalu.ch/coding/primes/list.php) primes between 2 and 10000. If you want to know how the query works, read on!

## Introduction

Many SQL databases offer an advanced functionality called "Recursive SQL Expressions". It allows you to query hierarchical data sets such as a family tree ("List all ancestors of a given person") or a movie-actor database ("What [Bacon](https://en.wikipedia.org/wiki/Six_Degrees_of_Kevin_Bacon) number does an actor have?"). It turns out, a more silly use case is to calculate prime numbers using the [Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes). Let's start with the basics:

*Definition:* A natural number is prime if it has exactly two positive divisors. The set can be written as

$$
\mathbb{P} = \lbrace n \in \mathbb{N} \ | \ d(n) = 2 \rbrace
$$

where $d(n)$ counts the number of divisors.

There are different ways to generate primes but one very straight forward way is the Sieve of Eratosthenes. The idea is to list all numbers from $2$ to $n$ and then crossing out numbers that are multiple of $2$, $3$, $5$, etc. until you reach $n$ (or rather $\sqrt{n}$). Everything that is left over will be prime. Implementing the sieve is a very common programming exercise when learning programming or during [interviews](https://leetcode.com/problems/count-primes/description/).

## Recursive SQL

[This](https://builtin.com/data-science/recursive-sql) article explains Recurisve SQL quite well. You start with an initial table $r_0$ (base case) and apply a query that modifies it somehow to get the partial result table $r_1$. When the new table isn't empty, you take $r_1$ as an input and generate $r_2$, and so on until the result becomes empty. The final result is then the union of all partial results $r = r_0 \cup r_1 \cup \dotsb \cup r_n$.

For example, consider the following recursive expression that will calculate powers of $2$ smaller than $100$.

```sql
with recursive power2(i) as (
  -- base case
  select 1 
  union
  -- recursion
  select i*2 from power2 where i*2 < 100
)

select i from power2 order by i ASC
```

It starts with $r_0 = 1$ and then doubles the previous result as long as the result is smaller than $100$. This is the stoping condition that prevents an inifinite loop of the query.

## Naive Implementation of the Sieve of Eratosthenes

Let's start with a simple implementation to explain the main idea. Let the column $i$ be an initial sequence from $2$ to $n$ and let the column $d$ be initially set to $2$. For every iteration, we take the last divisor $d$ and filter all numbers from $i$ that are divisible by $d$ (`mod(i, d) != 0`), except when $d$ divides itself (`i <= d`). Then we increase $d$ by one. All numbers that are left in the end will be prime numbers.

```sql
with recursive primes(i, d) as (
  select generate_series(2, 10000), 2
  union
  select i, d + 1
  from primes
  where d <= 10000 and (i <= d or mod(i, d) != 0)
)

select i
from primes
where d = 10000
order by primes.i ASC
```

Here is some visualization from iteration 0 to 1

|$i_0$  |$d_0$|→  |$i_1$  |$d_1$|Comment    |
|-------|-----------|---|-------|-----------|-----------|
|2      |2          |→  |2      |3          |2 is smaller/equal to 2
|3      |2          |→  |3      |3          |2 doesn't divide 3
|4      |2          |  |      |          |2 divides 4
|5      |2          |→  |5      |3          |2 doesn't divide 5
|6      |2          |  |      |          |2 divides 6|
||  |$\ldots$|||$\ldots$|
|9999   |2          |→  |9999   |3          |2 doesn't divide 9999|
|10000  |2          |   |      |          |2 divides 10000|

and from iteration 1 to 2

|$i_1$  |$d_1$|→  |$i_2$  |$d_2$|Comment    |
|-------|-----------|---|-------|-----------|-----------|
|2      |3          |→  |2      |4          |2 is smaller/equal to 3|
|3      |3          |→  |3      |4          |3 is smaller/equal to 3|
|5      |3          |→  |5      |4          |4 doesn't divide 5|
||  |$\ldots$|||$\ldots$|
|9999   |3          |   |       |           |3 divides 9999|

## Optimization

We can optimize the query a bit. Instead of testing every number from $d = 2$ to $d = 10000$, we only have to test numbers that were not multiples of previous divisors. For example, there is no need to check for $d = 4$ because $d = 2$ filters all multples of $4$ already. We can replace the term `d + 1` with `select min(i) from primes where i > d`. This already reduces 90 % of iterations in our case.

Another optimization is to let $d$ only run up to $\sqrt{n}$ and not $n$. Our sieve is essentially checking if a number is a factor of another number. Suppose all factors of a number $i$ are greater than $\sqrt{n}$. That means, that $i$ must be then greater than $n$ but we are only interested in numbers up to $n$.

This yields the final query which takes 7 ms on [HyPer DB](https://hyper-db.de/interface.html).

```sql
with recursive primes(i, d) as (
  select generate_series(2, 10000), 2
  union
  select i, (select min(i) from primes where i > d)
  from primes
  where d <= (select sqrt(max(i)) from primes) and (i <= d or mod(i, d) != 0)
)

select i
from primes
where d = (select max(d) from primes)
order by primes.i ASC
```