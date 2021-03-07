---
layout: post
title: "Python Features"
date: 2021-03-07 15:00:00 +0100
---

Collection of useful python features & syntactical sugar to improve code clarity

## VSCode setup

## Coding

### Type hints

### f-Strings

### Keyword-only arguments

force keywords

```python
def reverse_string(s, *, capitalize=False, remove_spaces=False, verbose=False):
    s = s.upper() if capitalize else s
    s = s.replace(" ", "") if remove_spaces else s
    rev_string = ''.join(reversed(s))
    if verbose:
        print(s, "->", rev_string)
    return rev_string

# reverse_string("Hello World", True, True)  # throws TypeError
reverse_string("Hello World", capitalize=True, remove_spaces=True)
```

### nonlocal

```python
def create_accumulator():
    current = 0
    def acc(number):
        nonlocal current
        current += number
        return current
    return acc

acc = create_accumulator()
print("Add 7:", acc(7))  # Add 7: 7
print("Add 3:", acc(3))  # Add 3: 10
```

### Extended Destructuring

```
a, b, c, *other, d = row
```


### Walrus Operator

```
if env_base := os.environ.get("PYTHONUSERBASE", None):
    return env_base
```

### inline if

### list comprehension