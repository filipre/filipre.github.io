---
title: "Getting a Royal Flush in Video Poker"
date: 2018-01-21
description: "What is the probability of getting a Royal Flush in GTA's \"Video Poker\" casino game?"
type: "post"
tags: ["math"]
weight: 1
---

Some months ago I came across the Twitch streamer [Joshimuz](https://twitch.tv/joshimuz) who has quite an interesting project: He tries to play through GTA San Andreas by solving *everything*. Not only he tries to achieve 100 % but he also set his own goals and challenges. At the same time, he gives insights into a lot of (speed running) techniques, interesting bugs and a little bit of game development. If you like content like that, you should definitely check out his video series [True 100%+](https://www.youtube.com/watch?v=FlOQslp4MQA) on Youtube.

Anyway, one of his goals is to get a Royal Flush in GTA's [video poker](https://en.wikipedia.org/wiki/Video_poker). Like in normal video poker, you get 5 cards and you decide which cards you want to keep. Then, you get new cards and depending on what kind of hand you have, you get some amount of money or nothing at all. Obviously, a Royal Flush gives you the most money but is also the least likely outcome. You can watch his first attempts [here](https://youtu.be/NU8m18HO35o?start=1468).

I wondered, how much does he have to play to get a Royal Flush? How likely (in terms of probability) is it if he uses the best strategy?

## tl;dr

By using the best strategy, the probability of getting a Royal Flush is about **0.0043 %** which means, *it's expected* that he plays around **23081** games of video poker.

Furthermore, we can provide the amount of games needed for getting a Royal Flush with a certain probability. For example: It's 50 % likely that Joshimuz gets a Royal Flush if he plays 15998 games. On the other hand, there is a 1 % chance that he might not get a Royal Flush even after 106288 games.

|p|n|
|-|-|
|1 %|232
|10 %|2432
|25 %|6640
|50 %|15998
|75 %|31996
|90 %|53144
|99 %|106288

If you are interested on how to derive these results, read on!

## Video Poker

Video poker uses a standard deck of 52 cards, i.e. the lowest cards are **2** :clubs:, **2** :spades:, **2** :hearts:, **2** :diamonds: and the highest cards are **A** :clubs:, **A** :spades:, **A** :hearts:, **A** :diamonds:. A Royal Flush is the highest street possible in the same suit. Therefore, there are four possible Royal Flushes, namely:

- **10** :clubs: **J** :clubs: **Q** :clubs: **K** :clubs: **A** :clubs:
- **10** :spades: **J** :spades: **Q** :spades: **K** :spades: **A** :spades:
- **10** :hearts: **J** :hearts: **Q** :hearts: **K** :hearts: **A** :hearts:
- **10** :diamonds: **J** :diamonds: **Q** :diamonds: **K** :diamonds: **A** :diamonds:

We will call these cards "potential cards".

For every video poker game, the player receives 5 random cards from the deck. He then can choose what he thinks are the best cards to keep and which cards should be thrown away. After that, the player receives new cards until he has 5 again and the dealer (i.e. the slot machine) evaluates the highest hand. In this problem we are not interested in getting any other winning hand like a "Full House" or "Three of a Kind". Notice that you can also adjust your wagger and that there are many varieties of video poker but this not really interesting to us either.

## Best Strategy

It is quite obvious that the best strategy for getting a Royal Flush is to keep potential cards (listed above) and to throw useless ones away. Sometimes, there are multiple potential cards but in different suits. Again, it's very easy to see that we should keep the suit that has more potential cards in that suit then the other. If the number is equal (for example if we have a hand like **10** :hearts: **J** :hearts: **K** :clubs: **A** :clubs: **5** :hearts:) then it does not matter if we keep :hearts: or :clubs: because both Royal Flushes come with the same probability.

In this post, we are not going to formally prove that this greedy strategy maximises our chances but I believe it should not be [too](https://web.stanford.edu/class/archive/cs/cs161/cs161.1138/handouts/120%20Guide%20to%20Greedy%20Algorithms.pdf) [hard](http://www.cs.cornell.edu/courses/cs482/2003su/handouts/greedy_exchange.pdf).

You can take a look at the implementation of this strategy in the `holdCards(hand)` method at the very end.

## Calculation of the Probability

Let $R$ denote the [event](https://en.wikipedia.org/wiki/Event_(probability_theory)) of a Royal Flush and $\Omega$ denote the set of all possible outcomes of drawing 5 cards from a standard deck. From the section above it's clear that there are $\vert R\vert = 4$ possible Royal Flushes. If we know the size of our sample space $\Omega$, i.e. how many ways there are to sample 5 cards from a 52 deck, we can easily calculate the probability by utilising this formula:

$$ \Pr(\text{event}) = \frac{\text{number of outcomes in event}}{\text{number of outcomes in sample space}} $$

Luckily, there exists the [Binomial Coefficient](https://en.wikipedia.org/wiki/Binomial_coefficient) $\binom{n}{k}$ that gives us the number of possibilities of choosing $k$ elements from a set of size $n$ without respecting the order. Here, we have $n = 52$ cards and we choose $k = 5$ cards from them. By employing the formula above we get

$$ \Pr(R) = \frac{|R|}{|\Omega|} = \frac{4}{\binom{52}{5}} = \frac{1}{649740} \approx 0.00015\% $$

However, our problem is a bit more complicated because we completely neglected that we can optimise our chances by keeping potential cards. So far we only calculated the probability of getting a Royal Flush when we are not allowed to keep cards. From now, $R$ will refer to the *actual* problem and not to this simple one.

### Law of Total Probability, Random Variables, ...

As it turns out, we need to divide our event $R$ into smaller disjoint subsets. The intuition behind doing that is that it is much more likely to get a Royal Flush if the player keeps 4 potential cards in the same suit than keeping only one or two. However, drawing 4 potential cards in the same suit in the first place is much less likely than getting maybe only one. By splitting up $R$ we can exactly describe this observation mathematically. This trick is known as the [Law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability):

$$ \Pr(A) = \sum_n \Pr(B_n) \cdot \Pr(A \;\vert\; B_n) $$

I will shortly explain what the bar inside $\Pr(A \;\vert\; B_n)$ means, but first, we also have to change our notation a bit by using [Random Variables](https://en.wikipedia.org/wiki/Random_variable). A Random Variable is usually a function $X\colon \Omega \to \mathbb{R}$ that maps events to natural numbers. That way, we can describe events more easily and work with them. Let $X$ be a Random Variable that denotes the number of cards a player exactly kept by using the described strategy. It's clear that the kept cards are all equal or above the rank **10** and all have the same suit. In addition, $X$ can only take values between 0 (not keeping any cards) and 5 (keeping all cards, aka. a Royal Flush). To refer to the probability of getting a Royal Flush *if the player already holds 3 cards*, we write

$$\Pr(R\;\vert\;X=3)$$

Probabilities that come with a *condition* are called [Conditional Probabilities](https://en.wikipedia.org/wiki/Conditional_probability) where the condition is written after the bar $\vert$. Here we used the notation involving the Random Variable $X$. If we want to refer to the probability of being able to keep three cards in the first place, we would write

$$\Pr(X=3)$$

Notice that it means something completely different. It should not be too hard to see that

$$0 < \Pr(R\;\vert\;X=0) < \Pr(R\;\vert\;X=1) < \dotsb < \Pr(R\;\vert\;X=5) = 1$$

but

$$1 > \Pr(X=0) > \Pr(X=1) > \dotsb > \Pr(X=5) > 0$$

Since we said that $X$ *exactly* represents the number of kept cards in a game, the associated events are disjoint and we can apply the Law of Total Probability:

$$\Pr(R) = \sum_{x=0}^5 \Pr(X=x) \cdot \Pr(R \;\vert\; X=x)$$

So if we find out how to calculate $\Pr(X=x)$ and $\Pr(R \;\vert\; X=x)$ for a given $x$, we are done! We will start with the easier factor $\Pr(R \;\vert\; X=x)$.

### Suppose we already hold cards...

Surely there are only $52-5$ cards left in deck, our hand has $x$ cards and we need to draw the remaining $5-x$ cards to have a full hand again. This sounds exactly like the situation at the very beginning but now, $n$ is $52-5=47$ and $k$ is $5-x$. Additionally, we need to think about the number of outcomes in the Royal Flush event $R$ depending on $x$. If we don't hold any card, we did not get any potential card and we could still draw any of the four possible Royal Flushes with the remaining $47$ cards in the deck. But, if we save at least one card, there is only one possible Royal Flush left since we sort of "decided" on a suit already. Therefore

$$\Pr(R \;\vert\; X=x) = \begin{cases}
   \frac{4}{\binom{52-5}{5}} &\text{if } x=0 \\
   \frac{1}{\binom{52-5}{5-x}} &\text{otherwise}
\end{cases}$$

It's always a good idea to test a model for some edge cases to detect errors. We would expect that if we keep $x=5$ cards, we *must* have a Royal Flush. Indeed this is true:

$\Pr(R \;\vert\; X=5) = \frac{1}{\binom{52-5}{5-5}} = \frac{1}{\binom{47}{0}} = \frac{1}{1} = 1$

Let's tackle $\Pr(X=x)$.

### One Random Variable is not enough

It looked easier than it was to be honest, because we have to make sure that we don't count some events twice or not at all. For example, suppose we have a hand (outcome) like this:

<center><strong>10</strong> :clubs: <strong>B</strong> :clubs: <strong>D</strong> :clubs: <strong>10</strong> :diamonds: <strong>B</strong> :diamonds:</center>
<br>

It's important that this outcome only *counts* when $x = 3$ and not $x = 2$. Otherwise, our subsets would not be disjoint and we could not apply the Law of Total Probability. But a lot of the simpler approaches and formulas just don't care about that and manually excluding situations like these is error prone and not convincing. This problem does not occur in the previous section because we throw cards away.

Instead, we split up $R$ now a bit differently to be able to use a more systematic approach. Let $X$ still be the Random Variable that keeps track of the number of cards a player wants to hold. Now, let $A$ be a Random Variable that notes the number of potential cards in the current hand *of a different suit* than the cards being tracked by $X$. In the hand above, we would have an event where $X=3$ and $A=2$. Let $B$ and $C$ be Random Variables too, where the former represents the third different suit and the latter represents the fourth different suit. Since the hand above does not have a third or fourth suit, we simply have $B=0$ and $C=0$. Also notice the number of non-potential cards is $5-X-A-B-C$, i.e. simply the rest.

By using only $X$, $A$, $B$ and $C$ we can describe any relevant event for us, but first, we must enforce additional constraints to make all events disjoint: First, we must not exceed 5 cards on a hand and second, there must be an ordering of the Random Variables because we have to make sure that we do not get accidentally a better hand by not saving $X$ cards.

1. Constraint: $X+A+B+C \le 5$
2. Constraint: $X \ge A \ge B \ge C \ge 0$

Let's give some examples to make it clearer.

|$X$|$A$|$B$|$C$|Example Hand|Valid?|
|-----|-----|-----|-----|------------|------|
|2|1|0|0|**6** :diamonds: **9** :hearts: **Q** :clubs: **K** :clubs: **K** :spades:|yes|
|1|1|1|1|**5** :clubs: **10** :clubs: **10** :hearts: **A** :diamonds: **A** :spades:|yes|
|5|0|0|0|**10** :hearts: **J** :hearts: **Q** :hearts: **K** :hearts: **A** :hearts:|yes|
|4|2|0|0|**10** :hearts: **J** :hearts: **Q** :hearts: **Q** :clubs: **K** :diamonds: **K** :hearts:|no, violates constraint 1|
|2|3|0|0|**10** :clubs: **J** :clubs: **Q** :spades: **K** :spades: **A** :spades:|no, violates constraint 2|

We need to rewrite $\Pr(R)$ though since we are dealing with new Random Variables now:

$$ \Pr(R) = \sum_{\scriptstyle x+a+b+c \le 5\atop\scriptstyle x \ge a \ge b \ge c \ge 0} \Pr(X=x, A=a, B=b, C=c) \cdot \Pr(R \;\vert\; X=x, A=a, B=b, C=c) $$

So do we have to throw $\Pr(R \;\vert\; X=x)$ away, because it does not take $A$, $B$ and $C$ into account? No we don't! Since we throw the cards tracked by $A$, $B$ and $C$ away *anyway*, the probability of getting a Royal Flush does not [depend](https://en.wikipedia.org/wiki/Independence_(probability_theory)) on them and we do have

$$ \Pr(R \;\vert\; X=x, A=a, B=b, C=c) = \Pr(R \;\vert\; X=x) $$

The big question is, how do we calculate $\Pr(X=x, A=a, B=b, C=c)$? To do this, again, we need to think about the number of outcomes in the event and the number of outcomes in the sample space.

### More combinatorics

The size of our sample space $\Omega$ is $\binom{52}{5}$ again since we start with a fresh deck and then draw 5 cards from it.

To determine the number of outcomes in the event $X=x, A=a, B=b, C=c$, we need to consider all different combinations of suits together with all different combinations of ranks. To make it easier first, let's fix the Random Variables to one specific suit: $X$ tracks :clubs:, $A$ tracks :spades:, $B$ tracks :hearts: and $C$ tracks :diamonds:. Let $r(x, a, b, c)$ represent the number of different rank combinations. If suits are fixed,

$$ \Pr(X=x, A=a, B=b, C=c) = \frac{r(x, a, b, c)}{\binom{52}{5}} $$

Finding a term for $r(x, a, b, c)$ is easy. Think about you have 5 different decks:

- 5 potential cards of suit :clubs:
- 5 potential cards of suit :spades:
- 5 potential cards of suit :hearts:
- 5 potential cards of suit :diamonds:
- 32 other (useless) cards of any suit

Now we draw $x$ cards from the :clubs: deck, $a$ cards from the :spades: deck, $b$ cards from the :hearts: deck, $c$ cards from the :diamonds: deck and the missing $5-x-a-b-c$ cards from the other cards:

$$ r(x, a, b, c) = \binom{5}{x} \cdot \binom{5}{a} \cdot \binom{5}{b} \cdot \binom{5}{c} \cdot \binom{32}{5-x-a-b-c} $$

Sadly (luckily?), our Random Variables can track any suit and it only matters, that two Random Variables do not track the same suit at the same time. Therefore, we have to calculate $r(x, a, b, c)$ for any possible suit as well. Let $s(x, a, b, c)$ denote the number of suit combinations. We then can simply multiply those two terms.

$$ \Pr(X=x, A=a, B=b, C=c) = \frac{ s(x, a, b, c) \cdot r(x, a, b, c)}{\binom{52}{5}} $$

Finding a nice term for $s(x, a, b, c)$ is a bit more tricky, but there are some observations: At first, we have four different possibilities to assign a suit to $X$. Because $A$ has to be a different suit than $X$, there are only three possibilities left. The same applies for $B$ and once we assigned suits to $X$, $A$ and $B$ already, there is only one possibility left for $C$. If our event consists of only two suits, we simply neglect the possibilities for the other two suits. In the equation below, $H(x)$ is a "left-continuous" [Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function) that is $0$ if $x$ is $0$ and is $1$ if $x>0$.

$$ s(x, a, b, c) = \frac{4^{H(x)} \cdot 3^{H(a)} \cdot 2^{H(b)} \cdot 1^{H(c)}}{g(x, a, b, c)} $$

So what is $g(x, a, b, c)$ doing? The enumerator *sometimes* counts suit combinations twice or more times but *sometimes* does not. On the one hand, the event $X=2, A=1, B=0, C=0$ has following combinations:

**[** :clubs: :clubs: :spades: **]**, **[** :clubs: :clubs: :hearts: **]**, **[** :clubs: :clubs: :diamonds: **]**, **[** :spades: :spades: :clubs: **]**, **[** :spades: :spades: :hearts: **]**, **[** :spades: :spades: :diamonds: **]**, **[** :hearts: :hearts: :clubs: **]**, **[** :hearts: :hearts: :spades: **]**, **[** :hearts: :hearts: :diamonds: **]**, **[** :diamonds: :diamonds: :clubs: **]**, **[** :diamonds: :diamonds: :spades: **]**, **[** :diamonds: :diamonds: :hearts: **]**

On the other hand, the event $X=1, A=1, B=0, C=0$ has much fewer:

**[** :clubs: :spades: **]**, **[** :clubs: :hearts: **]**, **[** :clubs: :diamonds: **]**, **[** :spades: :hearts: **]**, **[** :spades: :diamonds: **]**, **[** :hearts: :diamonds: **]**

If two/three/four Random Variables share the same number as others and are at least $1$ (here: $A = X = 1$), we need to remove additional counted outcomes due to permutations. We do this by using [factorials](https://en.wikipedia.org/wiki/Factorial). $n!$ is the number of ways how to permute $n$ elements. For instance $n=3$, we would get $n! = 3 \cdot 2 \cdot 1 = 6$ different ways to permute three suits:

**[** :clubs: :spades: :hearts: **]**, **[** :clubs: :hearts: :spades: **]**, **[** :spades: :clubs: :hearts: **]**, **[** :spades: :hearts: :clubs: **]**, **[** :hearts: :clubs: :spades: **]**, **[** :hearts: :spades: :clubs: **]**

The easiest way to find out $g(x, a, b, c)$ is to simply go through every valid assignment for $X=x, A=a, B=b, C=c$ and think about it directly. In many cases it is just 1.

|Example|$C$|$B$|$A$|$X$|$g$|
|-------------|-----|-----|-----|-----|-------|-------|-------|-------|-------|-----------------|
||0|0|0|0|1|
|:clubs:|0|0|0|1|1|
|:clubs: :spades:|0|0|1|1|2|
|:clubs: :spades: :hearts:|0|1|1|1|6|
|:clubs: :spades: :hearts: :diamonds:|1|1|1|1|24|
|:diamonds: :diamonds:|0|0|0|2|1|
|:diamonds: :diamonds: :clubs:|0|0|1|2|1|
|:diamonds: :diamonds: :clubs: :spades:|0|1|1|2|2|
|:diamonds: :diamonds: :clubs: :spades: :hearts:|1|1|1|2|6|
|:diamonds: :diamonds: :hearts: :hearts:|0|0|2|2|2|
|:diamonds: :diamonds: :spades: :spades: :clubs:|0|1|2|2|2|
|:spades: :spades: :spades:|0|0|0|3|1|
|:spades: :spades: :spades: :diamonds:|0|0|1|3|1|
|:spades: :spades: :spades: :diamonds: :clubs:|0|1|1|3|2|
|:spades: :spades: :spades: :diamonds: :diamonds:|0|0|2|3|1|
|:hearts: :hearts: :hearts: :hearts:|0|0|0|4|1|
|:hearts: :hearts: :hearts: :hearts: :clubs:|0|0|1|4|1|
|:clubs: :clubs: :clubs: :clubs: :clubs:|0|0|0|5|1|

<!--
But, you can also come up with a general formula which I found more confusing than explaining at the end though:

Let $l_i(x, a, b, c)$ represent the amount of variables $x$, $a$, $b$ and $c$ taking value $i$. That means, $l_i(x, a, b, c)$ is between $0$ and $4$ since there are only four suits anyway. Because our Random Variables range between 0 and 5, we keep books of $l_1$ to $l_5$ in the table below. $l_0$ is not needed.

To calculate how many suit permutations there are for given $x$, $a$, $b$ and $c$, we define $g(x, a, b, c)$

$$g(x, a, b, c) = \prod_{i=1}^5 l_i(x, a, b, c)! $$

The multiplication is needed since it *could* happen that multiple $l_i$s are equal/greater than $2$ and then we would have to remove permutations not only once but multiple times. It turned out this is not the case for Video Poker. But think of this situation if there were six cards instead of only five: :clubs: :clubs: :spades: :spades: :hearts: :diamonds:.

|Example|$C$|$B$|$A$|$X$|$l_1$|$l_2$|$l_3$|$l_4$|$l_5$|$g$|
|-------------|-----|-----|-----|-----|-------|-------|-------|-------|-------|-----------------|
||0|0|0|0|0|0|0|0|0|1|
|:clubs:|0|0|0|1|1|0|0|0|0|1|
|:clubs: :spades:|0|0|1|1|**2**|0|0|0|0|2|
|:clubs: :spades: :hearts:|0|1|1|1|**3**|0|0|0|0|6|
|:clubs: :spades: :hearts: :diamonds:|1|1|1|1|**4**|0|0|0|0|24|
|:diamonds: :diamonds:|0|0|0|2|0|1|0|0|0|1|
|:diamonds: :diamonds: :clubs:|0|0|1|2|1|1|0|0|0|1|
|:diamonds: :diamonds: :clubs: :spades:|0|1|1|2|**2**|1|0|0|0|2|
|:diamonds: :diamonds: :clubs: :spades: :hearts:|1|1|1|2|**3**|1|0|0|0|6|
|:diamonds: :diamonds: :hearts: :hearts:|0|0|2|2|0|**2**|0|0|0|2|
|:diamonds: :diamonds: :spades: :spades: :clubs:|0|1|2|2|1|**2**|0|0|0|2|
|:spades: :spades: :spades:|0|0|0|3|0|0|1|0|0|1|
|:spades: :spades: :spades: :diamonds:|0|0|1|3|1|0|1|0|0|1|
|:spades: :spades: :spades: :diamonds: :clubs:|0|1|1|3|**2**|0|1|0|0|2|
|:spades: :spades: :spades: :diamonds: :diamonds:|0|0|2|3|0|1|1|0|0|1|
|:hearts: :hearts: :hearts: :hearts:|0|0|0|4|0|0|0|1|0|1|
|:hearts: :hearts: :hearts: :hearts: :clubs:|0|0|1|4|1|0|0|1|0|1|
|:clubs: :clubs: :clubs: :clubs: :clubs:|0|0|0|5|0|0|0|0|1|1|
-->

We are finally in a position where we can calculate $\Pr(X=x, A=a, B=b, C=c)$! Let's do it:

|Example|$C$|$B$|$A$|$X$|$\Pr(X=x, A=a, B=b, C=c)$|
|-------|-----|-----|-----|-----|-------------------------|
||0|0|0|0|7.75 %|
|:clubs:|0|0|0|1|27.67 %|
|:clubs: :spades:|0|0|1|1|28.63 %|
|:clubs: :spades: :hearts:|0|1|1|1|9.54 %|
|:clubs: :spades: :hearts: :diamonds:|1|1|1|1|0.77 %|
|:diamonds: :diamonds:|0|0|0|2|7.63 %|
|:diamonds: :diamonds: :clubs:|0|0|1|2|11.45 %|
|:diamonds: :diamonds: :clubs: :spades:|0|1|1|2|3.69 %|
|:diamonds: :diamonds: :clubs: :spades: :hearts:|1|1|1|2|0.19 %|
|:diamonds: :diamonds: :hearts: :hearts:|0|0|2|2|0.74 %|
|:diamonds: :diamonds: :spades: :spades: :clubs:|0|1|2|2|0.23 %|
|:spades: :spades: :spades:|0|0|0|3|0.76 %|
|:spades: :spades: :spades: :diamonds:|0|0|1|3|0.74 %|
|:spades: :spades: :spades: :diamonds: :clubs:|0|1|1|3|0.12 %|
|:spades: :spades: :spades: :diamonds: :diamonds:|0|0|2|3|0.05 %|
|:hearts: :hearts: :hearts: :hearts:|0|0|0|4|0.02 %|
|:hearts: :hearts: :hearts: :hearts: :clubs:|0|0|1|4|0.01 %|
|:clubs: :clubs: :clubs: :clubs: :clubs:|0|0|0|5|0.0002 %|
|**Sum**|||||100 %|

We quickly confirm our calculation by summing up all $p(X=x, A=a, B=b, C=c)$. If we would not get $1$ back, that would mean some events count outcomes too much or not at all.

$$ \sum_{\scriptstyle x+a+b+c \le 5\atop\scriptstyle x \ge a \ge b \ge c \ge 0} p(X=x, A=a, B=b, C=c) = 1 $$

We are almost done.

### Putting everything together

Since I also wanted to know $\Pr(X=x)$, I simply added all disjoint subset satisfying $X=x$ together, like this:

$$\Pr(X=4) = \Pr(X=4, A=0, B=0, C=0) + \Pr(X=4, A=1, B=0, C=0)$$

These are my results

|Example|$X$|$\Pr(X=x)$|$\Pr(R \;\vert\; X=x)$|
|-------|-----|----------|----------------------|
||0|7.75 %|0.0003 %|
|:diamonds:|1|66.61 %|0.001 %|
|:clubs: :clubs:|2|23.94 %|0.01 %|
|:hearts: :hearts: :hearts:|3|1.66 %|0.09 %|
|:spades: :spades: :spades: :spades:|4|0.04 %|2.13 %|
|:diamonds: :diamonds: :diamonds: :diamonds: :diamonds:|5|0.0002 %|100 %|
|**Sum**||100 %||

Finally, by using Law of Total Probability:

$$ \Pr(R) = \sum_{x=0}^5 \Pr(X=x) \cdot \Pr(R \;\vert\; X=x) \approx \frac{1}{23081} \approx 0.0043 \% $$

### Number of expected games and more

Let $E$ denote the expected number of games we need to play until we get a Royal Flush. If we get one, we stop. Otherwise we try again. Read [this](https://math.stackexchange.com/questions/42930/what-is-the-expected-value-of-the-number-of-die-rolls-necessary-to-get-a-specifi) for details.

$$ E = 1 + \frac{23080}{23081} \cdot E \Rightarrow E = 23081$$

This does not tell us that much because it rather means, *on average* we need 23081 games. However, I am pretty sure we don't want to get Royal Flushes a second or third time. There is something else we can do: We could provide a probability $p$ on how likely it is that we get a Royal Flush in the first $n$ tries.

With the help of [complementary events](https://en.wikipedia.org/wiki/Complementary_event) we solve for $n$:

$$p = 1 - (1-\Pr(R))^n \Leftrightarrow 1-p = (1-\Pr(R))^n \Leftrightarrow \ln{(1-p)} = n \ln{(1-\Pr(R))}$$

$ \Leftrightarrow n = \frac{\ln(1-p)}{\ln(1-\Pr(R))} $

I think this table describes much better how much effort Josh might have to put into his project.

|p|n|
|-|-|
|1%|232
|10%|2432
|25%|6640
|50%|15998
|75%|31996
|90%|53144
|99%|106288

## Empirical Validation

To confirm the theory, I wrote a small Python script that simulates getting a Royal Flush 10000 times. After 4h on my MacBook, the probability was $0.0042\%$, which is not too far away from $0.0043 \%$. You can find the code below:

```python
from pcards import Deck, Card
from itertools import chain, combinations
from joblib import Parallel, delayed
import multiprocessing
from statistics import mean

# https://docs.python.org/2.7/library/itertools.html#recipes
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def printHand(hand):
    [print(card) for card in hand]
    print()

def drawInitial(deck):
    return deck.draw(5)

def holdCards(hand):
    high_cards = [card for card in hand if card.rank() >= 10]
    handPowerset = [list(subset) for subset in powerset(high_cards)]
    bestHand = []
    for potentialHand in handPowerset:
        if len(potentialHand) <= 0:
            continue
        suit = potentialHand[0].suit()
        if all(card.suit() == suit for card in potentialHand):
            if len(potentialHand) > len(bestHand):
                bestHand = potentialHand
    return bestHand

def drawMissing(deck, hand):
    return hand + deck.draw(5 - len(hand))

def isRoyalFlush(hand):
    if all(card.rank() >= 10 for card in hand):
        suit = hand[0].suit()
        if all(card.suit() == suit for card in hand):
            return True
    return False

def step(deck):
    firstDraw = drawInitial(deck)
    holdedCards = holdCards(firstDraw)
    secondDraw = drawMissing(deck, holdedCards)
    return secondDraw

def getNumberOfTries(job):
    i = 0
    while True:
        i = i + 1
        deck = Deck()
        deck.shuffle()
        hand = step(deck)
        if isRoyalFlush(hand):
            print(job, "done")
            break
    return i

def runSimulation(n = 100, numCores = multiprocessing.cpu_count()):
    print("Run", n, "simulations on", numCores, "cores:")
    results = Parallel(n_jobs=numCores)(delayed(getNumberOfTries)(i) for i in range(0, n))
    print(mean(results))

runSimulation(n = 10000)
```

*Update*: In case you come up with a better way, please let me know.

*Update*: [@MrSmithVP](https://twitter.com/MrSmithVP) simulated this already [here](https://gtaforums.com/topic/886791-video-poker/) using C++ and a more efficient implementation.

*Update*: Joshimuz mentioned my blog post in one of his [videos](https://youtu.be/S4Wb8EFN1fU?t=2547). Thanks!

*Update*: "He did it!" See comments or [this](https://reddit.com/r/LivestreamFail/comments/a6opyj/royal_flush_in_gtasa/) Reddit thread.
