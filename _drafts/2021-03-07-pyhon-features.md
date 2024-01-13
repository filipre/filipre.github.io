---
layout: post
title: "Python Features"
date: 2021-03-07 15:00:00 +0100
---

Collection of useful python features & syntactical sugar to improve code clarity

Idea: use *interesting* math examples

## Sources

- https://betterprogramming.pub/5-advanced-python-concepts-explanations-and-applications-99a03f6bd1bd
- 

## VSCode setup / Debugging

### pprint

```python
from pprint import pprint
```


## Coding

### f-Strings

```python
print(f"")
```

### inline if

### for-else

### list comprehension

### *args and **kwargs

### any, all

### Map, Filter and Reduce

### Type hints




### Keyword-only arguments

force keywords

```python
def reverse_string(s, *, capitalize=False, remove_spaces=False, verbose=False):
    s = s.upper() if capitalize else s
    s = s.replace(" ", "") if remove_spaces else s
    rev_string = s[::-1]
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

### Lambdas

### Comprehensions

### Context Managers (`with` statement)

https://book.pythontips.com/en/latest/context_managers.html


### Decorators

### Generators



### Walrus Operator
see https://www.python.org/dev/peps/pep-0572/
```python
if env_base := os.environ.get("PYTHONUSERBASE", None):
    return env_base
```


### Co-routines

### Collections (defaultdict, deque, Counter)

## Magic Methods

- `__len__`
- `__getitem__`


## Memory Management

Todo: watch that Youtube video again that explains counters and garbage collection



