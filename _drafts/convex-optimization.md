# Convex Optimization Lecture Notes

## Convex Analysis

### Convex Set

**Definition** A set $C$ is convex if for all $u, v \in C$ and for all $\alpha \in [0, 1]$:

$$\alpha u + (1-\alpha) v \in C$$

**Definition** A set $C \subset \mathbb{E}$ is open if for all $u \in C$ there exists a $\epsilon > 0$ such that $B_\epsilon(u) \subset C$ where $B_\epsilon(u) :=  \{ v \in \mathbb{E} : \| v - u \| < \epsilon \}$.

**Definition** A set $C \subset \mathbb{E}$ is closed if $\mathbb{E} \ C$ is open.
