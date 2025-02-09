---
title: 'Todo: Trees'
date: 2024-06-16T10:05:00+01:00
description: "Todo"
type: "post"
tags: ["algorithms", "interview"]
draft: true # weight: 2
---

Talk about e18e and https://deptree.rschristian.dev/?q=eslint-plugin-import
connected acyclic undirected graph

## Tree Representation

Binary Tree:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

Any tree:

```python
class TreeNode:
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children if children else []
```

## Tree Traversal

https://leetcode.com/problems/serialize-and-deserialize-binary-tree

Depth First Search

```python
def visit(root: Optional[TreeNode]):
    if root is None:
        return
    visit(root.left)
    print(root.val)
    visit(root.right)
```

Breath First Search

## Tree Serialization

Serialize

```python
def serialize(root: Optional[TreeNode]):
    res = []

    def dfs(node: TreeNode):
        if node is None:
            return
        res.append(str(node.val))
        for child in node.children:
            dfs(child)
            res.append("^")

    dfs(root)

    return ",".join(res)
```

Deserialize

```python
def deserialize(s: str):
    chars = s.split(",")
    if not chars:
        return None

    curr = TreeNode(-1)  # used for anchoring
    ancestors = []

    for c in chars:
        if c != "^":
            node = TreeNode(int(c))
            curr.children.append(node)
            ancestors.append(curr)
            curr = node
        else:
            curr = ancestors.pop()

    return curr
```

## Number of unique Subtrees

Simple: Value is unique, then you can just count values
Otherwise:

```python
def countSubtrees(root: TreeNode):
    res = defaultdict(int)

    def dfs(node: Optional[TreeNode]):
        if node is None:
            pass

        subtree = [str(node.val)]
        for child in node.children:
            childtree = dfs(child)
            for nodes in childtree:
                subtree.append(nodes)
            subtree.append("^")

        res[",".join(subtree)] = res[",".join(subtree)] + 1

        return subtree

    dfs(root)

    return res
```