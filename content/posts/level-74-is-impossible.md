---
title: 'Level 74 Is Impossible'
date: 2024-03-04T23:02:39+01:00
description: "Writing a simple backtracking algorithm to convince myself that the game I was playing cannot be beaten"
type: "post"
tags: ["algorithms", "interview"]
weight: 2
---

A while ago I downloaded a simple puzzle game called [Water Sorting Puzzle](https://play.google.com/store/apps/details?id=com.maddin.watersortpuzzle&hl=gsw&gl=US&pli=1) where you have multiple flasks with different colors of liquid and you have to somehow sort them to beat a level. The nice thing about this app is that it's completly ad free - unlike many other apps with the same game concept. I was playing it until level 74 where I hit a wall. Feel free to try it out yourself:

![Level 74](/posts/level-74-is-impossible/level74.png)

As it turns out, the level is impossible. To show this, I wrote a backtracking algorithm that iterates through every possible state and "backtracks" when flasks cannot be sorted further. I captured the results for the level above using [asciinema](https://asciinema.org/a/Nxv6030FYkxbjaHStt8EgUQNC)

{{< unsafe >}}
<script async id="asciicast-Nxv6030FYkxbjaHStt8EgUQNC" src="https://asciinema.org/a/Nxv6030FYkxbjaHStt8EgUQNC.js"></script>
{{< /unsafe >}}

The code can be found on my [Github](https://github.com/filipre/color-sorter). Here is the basic idea of backtracking written as a handy template for other problems:

```python
def solve(state) -> bool:
    if finished(state):
        return True
    
    for option in possibilities:
        apply(option, state)

        if solve(state):
            return True

        undo(option, state)

    return False
```

In our case, different operations can lead to the same state so you also need to keep track of game states that you've seen already. Otherwise you run into infinite loops. Conveniently, the emoji visualization above can be used to hash a state and store it in a hash table.

By the way, the app does allow you to add additional flasks though it's not obvious when you absolutly have to add flasks and when you just make the puzzle a bit easier. Here is a possible solution when you use 13 flasks instead of 12.

{{< details "Spoiler: Solution" >}}
```
Pour 0 into 11
Pour 0 into 12
Pour 0 into 13
Pour 0 into 11
Pour 2 into 0
Pour 2 into 12
Pour 2 into 13
Pour 3 into 12
Pour 4 into 2
Pour 6 into 4
Pour 6 into 11
Pour 6 into 12
Pour 8 into 0
Pour 7 into 8
Pour 6 into 7
Pour 1 into 6
Pour 1 into 3
Pour 3 into 1
Pour 8 into 6
Pour 1 into 3
Pour 8 into 13
Pour 5 into 8
Pour 4 into 5
Pour 10 into 8
Pour 2 into 10
Pour 5 into 4
Pour 10 into 2
Pour 7 into 10
Pour 6 into 7
Pour 7 into 6
Pour 9 into 7
Pour 0 into 9
Pour 9 into 0
Pour 9 into 11
Pour 0 into 9
Pour 1 into 0
Pour 1 into 7
Pour 3 into 0
Pour 2 into 3
Pour 4 into 1
Pour 1 into 5
Pour 4 into 13
Pour 5 into 1
Pour 5 into 2
Pour 5 into 7
Pour 10 into 2
```
{{< /details >}}

My guess is that the app generates puzzles on the fly and that I just got unlucky. I will write [Martin Kunze](https://martinkunze.com/) and ask, if my analysis is correct. Let's see!