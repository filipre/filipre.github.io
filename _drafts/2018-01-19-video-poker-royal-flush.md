---
layout: post
title:  "How Likely is a Royal Flush in GTA's Video Poker?"
date:   2018-01-19 21:24:00 +0900
published: true
---

**tl;dr version below**

Some months ago I came across the Twitch streamer [Joshimuz](https://twitch.tv/joshimuz) who has a quite interesting project: He tries to play through GTA San Andreas by solving *everything*. Not only he tries to archieve 100 % but he also set his own goals and challenges. At the same time, he gives insights into a lot (speed running) techniques, interesting bugs and a little bit of game development. If you like content like that, you should definitely check out his video series [True True 100%+](https://www.youtube.com/watch?v=FlOQslp4MQA) on Youtube.

Anyway, one of his goals is to get a Royal Flush in GTA's [video poker](https://en.wikipedia.org/wiki/Video_poker). Like in normal video poker, you get 5 cards and you decide, which cards you want to keep. Then, you get new cards and depending on what kind of hand you have, you get some amount of money or nothing at all. Obviously, a Royal Flush gives you the most money but is also the least likely outcome. You can watch his first attempts here:

{% youtube "https://youtu.be/NU8m18HO35o?start=1468" %}
<br>

I wondered, how much does he have to play to get a Royal Flush? How likely (in terms of probability) is it if he uses the best strategy?

## tl;dr

By using the best strategy, the probability of getting any royal flush is about **0.004 %** which means, *it's expected* that he plays around **23081** games of video poker.

Furthermore, we can provide the amount of games needed for getting a Royal Flush with a certain probability. For example: It's 50 % likely that Joshimuz gets a Royal Flush if he plays 15998 games.

|p|n|
|-|-|
|1 %|232
|10 %|2432
|25 %|6640
|50 %|15998
|75 %|31996
|90 %|53144
|99 %|106288

If you are interested on how to get to these results, read on!

## Video Poker

Video poker uses a standard deck of 52 cards, i.e. the lowest cards are **2** :clubs:, **2** :spades:, **2** :hearts:, **2** :diamonds: and the highest cards are **A** :clubs:, **A** :spades:, **A** :hearts:, **A** :diamonds:. A Royal Flush is the highest street possible in the same suit. Therefore, there are four possible Royal Flushes, namely:

- **10** :clubs:, **B** :clubs:, **D** :clubs:, **K** :clubs:, **A** :clubs:
- **10** :spades:, **B** :spades:, **D** :spades:, **K** :spades:, **A** :spades:
- **10** :hearts:, **B** :hearts:, **D** :hearts:, **K** :hearts:, **A** :hearts:
- **10** :diamonds:, **B** :diamonds:, **D** :diamonds:, **K** :diamonds:, **A** :diamonds:

We will call these cards "potential cards".

For every video poker game, the player receives 5 random cards from the deck. He then can choose what he thinks are the best cards to keep and which cards should be thrown away. After that, the player receives new cards until he has 5 again and the dealer (i.e. the slot machine) evaluates the highest hand. We are not interested in getting any other winning hand like a "Full House" or "Three of a Kind".

Notice that you can also adjust your wagger and that there are many varieties of video poker but this not really interesting to us.

## Best Strategy

It is quite obvious that the best strategy for getting a Royal Flush is to keep potential cards (listed above) and to throw useless ones away. Sometimes, there are multiple potential cards but in different suits. Again, it's very easy to see that we should keep the suit that has more potential cards in that colour then the other. If the number is equal (for example if we have a hand like **10** :hearts:, **B** :hearts:, **K** :clubs:, **A** :clubs:, **5** :hearts:) then it does not matter, if we keep :hearts: or :clubs: because both Royal Flushes come with the same probability.

In this post, we are not going to formally prove that this greedy strategy maximises our chances but I believe it should not be [too](https://web.stanford.edu/class/archive/cs/cs161/cs161.1138/handouts/120%20Guide%20to%20Greedy%20Algorithms.pdf) [hard](http://www.cs.cornell.edu/courses/cs482/2003su/handouts/greedy_exchange.pdf).

You can take a look at the implementation of this strategy in the `holdCards(hand)` method below.

## Calculation of the Probability

Let $$R$$ denote the [event](https://en.wikipedia.org/wiki/Event_(probability_theory)) of a Royal Flush and $$\Omega$$ denote the set of all possible outcomes of drawing 5 cards from a standard deck. From the section above it's clear that there are $$\vert R\vert = 4$$ possible Royal Flushes. If we know the size of our sample space $$\Omega$$, i.e. how many ways there are to sample 5 cards from a 52 deck, we can easily calculate the probability by utilising this formula:

$$ \Pr(\text{event}) = \frac{\text{number of outcomes in event}}{\text{number of outcomes in sample space}} $$

Luckily, there exists the [Binomial Coefficient](https://en.wikipedia.org/wiki/Binomial_coefficient) $$\binom{n}{k}$$ that gives us the number of possibilities of choosing $$k$$ elements from a set of size $$n$$ without respecting the order. Here, we have $$n = 52$$ cards and we choose $$k = 5$$ cards from them. By employing the formula above we get

$$ \Pr(R) = \frac{|R|}{|\Omega|} = \frac{4}{\binom{52}{5}} = \frac{1}{649740} \approx 0.00015\% $$

However, our problem is a bit more complicated because we completely neglected that we can optimise our chances by keeping potential cards. So far we only calculated the probability of getting a Royal Flush when we are not allowed to keep cards. From now, $$R$$ will refer to the *actual* problem and not to the simpler one above.

### Law of Total Probability, Random Variables, and more

As it turns out, we need to divide our event $$R$$ into smaller disjoint subsets. The intuition behind doing this is that it is much more likely to get a Royal Flush if the player keeps 4 potential cards in the same suit than keeping only one or two. However, drawing 4 potential cards in the same suit in the first place is much less likely than getting maybe only one. By splitting up $$R$$ we can exactly describe this observation mathematically. This trick is known as the [Law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability):

$$ \Pr(A) = \sum_n \Pr(B_n) \cdot \Pr(A \;\vert\; B_n) $$

I will shortly explain what the bar inside $$\Pr(A \;\vert\; B_n)$$ means but first, we also have to change our notation a bit by using [Random Variables](https://en.wikipedia.org/wiki/Random_variable). A Random Variable is usually a function $$X\colon \Omega \to \mathbb{R}$$ that maps events to natural numbers. That way, we can describe more easily events and work with them. Let $$X$$ be a Random Variable that denotes the number of cards a player exactly kept by using the described strategy. It's clear that the kept cards are all equal or above the rank **10** and all have the same suit. For example, to refer to the probability of getting a Royal Flush *if the player already holds 3 cards* we write

$$\Pr(R\;\vert\;X=3)$$

Probabilities that come with a *condition* are called [Conditional Probabilities](https://en.wikipedia.org/wiki/Conditional_probability) where the condition is written after the bar $$\vert$$. Here we used the notation involving the Random Variable $$X$$. If we want to refer to the probability of being able to keep three cards in the first place, we would write

$$\Pr(X=3)$$

Notice that it means something completely different. It should not be too hard to see that

$$0 < \Pr(R\;\vert\;X=0) < \Pr(R\;\vert\;X=1) < \dotsb < \Pr(R\;\vert\;X=5) = 1$$

but

$$1 > \Pr(X=0) > \Pr(X=1) > \dotsb > \Pr(X=5) > 0$$

Since we said that $$X$$ *exactly* represents the number of kept cards in a game, the associated events are disjoint and we can apply the Law of Total Probability:

$$\Pr(R) = \sum_{x=0}^5 \Pr(X=x) \cdot \Pr(R \;\vert\; X=x)$$

So if we find out how to calculate $$\Pr(X=x)$$ and $$\Pr(R \;\vert\; X=x)$$ for a given $$x$$, we are done! We will start with the easier factor $$\Pr(R \;\vert\; X=x)$$.

### Suppose we already hold $$x$$ cards...

asdfasdfaaaaaaa

$$p(R \;\vert\; X=x) = \begin{cases}
   4 \cdot \binom{52-5}{5} &\text{if } x=0 \\
   1 \cdot \binom{52-5}{5-x} &\text{otherwise}
\end{cases} $$

adsfadfs

$$ p(R) = \sum_{\scriptstyle a+b+c+d \le 5\atop\scriptstyle 0 \le a \le b \le c \le d} p(A=a, B=b, C=c, D=d) \cdot p(R \;\vert\; A=a, B=b, C=c, D=d) $$

test

$$ p(A=a, B=b, C=c, D=d) = \frac{ s(a, b, c, d) \cdot r(a, b, c, d)}{\binom{52}{5}} $$

TODO: longest sequence of 1, 2, 3, 4, 5 table

$$ s(a, b, c, d) = \frac{1^{H(a)} \cdot 2^{H(b)} \cdot 3^{H(c)} \cdot 4^{H(d)}}{g(a, b, c, d)} $$

...

$$g(a, b, c, d) = \prod_{i=1}^5 l_i(a, b, c, d)! $$

...

|Example Event|$$A$$|$$B$$|$$C$$|$$D$$|$$l_1$$|$$l_2$$|$$l_3$$|$$l_4$$|$$l_5$$|$$g$$|
|-------------|-----|-----|-----|-----|-------|-------|-------|-------|-------|-----------------|
||0|0|0|0|0|0|0|0|0|1|
|:clubs:|0|0|0|1|1|0|0|0|0|1|
|:clubs: :spades:|0|0|1|1|2|0|0|0|0|2|
|:clubs: :spades: :hearts:|0|1|1|1|3|0|0|0|0|6|
|:clubs: :spades: :hearts: :diamonds:|1|1|1|1|4|0|0|0|0|24|
|:diamonds: :diamonds:|0|0|0|2|0|1|0|0|0|1|
|:diamonds: :diamonds: :clubs:|0|0|1|2|1|1|0|0|0|1|
|:diamonds: :diamonds: :clubs: :spades:|0|1|1|2|2|1|0|0|0|2|
|:diamonds: :diamonds: :clubs: :spades: :hearts:|1|1|1|2|3|1|0|0|0|6|
|:diamonds: :diamonds: :hearts: :hearts:|0|0|2|2|0|2|0|0|0|2|
|:diamonds: :diamonds: :spades: :spades: :clubs:|0|1|2|2|1|2|0|0|0|2|
|:spades: :spades: :spades:|0|0|0|3|0|0|1|0|0|1|
|:spades: :spades: :spades: :diamonds:|0|0|1|3|1|0|1|0|0|1|
|:spades: :spades: :spades: :diamonds: :clubs:|0|1|1|3|2|0|1|0|0|2|
|:spades: :spades: :spades: :diamonds: :diamonds:|0|0|2|3|0|1|1|0|0|1|
|:hearts: :hearts: :hearts: :hearts:|0|0|0|4|0|0|0|1|0|1|
|:hearts: :hearts: :hearts: :hearts: :clubs:|0|0|1|4|1|0|0|1|0|1|
|:clubs: :clubs: :clubs: :clubs: :clubs:|0|0|0|5|0|0|0|0|1|1|

text...

$$ r(a, b, c, d) = \binom{5}{a} \cdot \binom{5}{b} \cdot \binom{5}{c} \cdot \binom{5}{d} \cdot \binom{32}{5-a-b-c-d} $$

test

$$ p(R \;\vert\; A=a, B=b, C=c, D=d) = p(R \;\vert\; D=d) = \begin{cases}
   4 \cdot \binom{52-5}{5} &\text{if } d=0 \\
   1 \cdot \binom{52-5}{5-d} &\text{otherwise}
\end{cases} $$

test

blabla

|Example Event|$$A$$|$$B$$|$$C$$|$$D$$|$$p(A=a, B=b, C=c, D=d)$$|
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

...

Notice that

$$ \sum_{\scriptstyle a+b+c+d \le 5\atop\scriptstyle 0 \le a \le b \le c \le d} p(A=a, B=b, C=c, D=d) = 1 $$

...

|Example Event|$$D$$|$$p(D=d)$$|$$p(R \;\vert\; D=d)$$|
|-------|-----|----------|----------------------|
||0|7.75 %|0.0003 %|
|:diamonds:|1|66.61 %|0.001 %|
|:clubs: :clubs:|2|23.94 %|0.01 %|
|:hearts: :hearts: :hearts:|3|1.66 %|0.09 %|
|:spades: :spades: :spades: :spades:|4|0.04 %|2.13 %|
|:diamonds: :diamonds: :diamonds: :diamonds: :diamonds:|5|0.0002 %|100 %|

more text. Again $$\sum_{d=0}^5 p(D=d) = 1$$

Explain Expectation: https://math.stackexchange.com/questions/42930/what-is-the-expected-value-of-the-number-of-die-rolls-necessary-to-get-a-specifi

|Event|$$p(R)$$|Expectation $$\mathbb{E} N(p(R))$$|
|-------|----|----------------------------------|
|$$R$$: Any Royal Flush|0.004 %|23081|

...

### TODO

$$p = 1 - (1-\Pr(R))^n \Rightarrow 1-p = (1-\Pr(R))^n \Rightarrow \ln{(1-p)} = n * \ln{(1-\Pr(R))}$$

$$ n = \frac{\ln(1-p)}{\ln(1-\Pr(R))} $$

## Empirical Validation

python... github...

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

**Time**: 4h
**Result**: 24029.4672
