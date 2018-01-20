---
layout: post
title:  "Video Poker Royal Flush"
date:   2018-01-19 21:24:00 +0900
published: true
---

What are the odds?

TODO: Introduction

{% youtube "https://youtu.be/NU8m18HO35o?start=1468" %}

...


## tl;dr

|Event|$$p(R)$$|Expectation $$\mathbb{E} N(p(R))$$|Epirical Result|
|-------|----|----------------------------------|---------------|
|$$R$$: Any Royal Flush|0.004 %|23081|...|

## Video Poker

...

## Best Strategy

TODO

## Theoretical Result

Short intro into Probability theory

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
