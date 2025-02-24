+++
date = '2025-02-23T13:34:46+01:00'
draft = true
title = 'Supernash'
slug = 'supernash'
+++

# Supernash Project

This figure represents the main method of the Supernash Project. One finds a Nash equilibrium in the Memory 1 Infinitely Repeated Game by finding a strategy that puts the blue dot into the dark region in the plane.

## Introduction

We consider the infinitely repeated game. In each round, players can choose either to **cooperate** or **defect**. A strategy belongs to the Memory 1 space if it conditions its next move solely on the previous round. In our case, the strategy is parameterized by four probabilities representing the outcomes of the last round:

**{CC, CD, DC, DD}**

where the first character is the move of the first player and the second is that of the second player. These probabilities form the strategy **{p<sub>CC</sub>, p<sub>CD</sub>, p<sub>DC</sub>, p<sub>DD</sub>}**.

## Goal

We define a Nash strategy *x* as one that satisfies:

**π(y, x) < π(x, x)**

for all strategies *y*, where **π(y, x)** is the long-run average payoff of strategy *y* against strategy *x*. It is given by the long-run frequencies of the repeated game:

**π(x, y) = R · v<sub>CC</sub> + S · v<sub>CD</sub> + T · v<sub>DC</sub> + P · v<sub>DD</sub>**

At first, checking every possible strategy *y* against a candidate strategy *x* seems daunting. However, a neat trick simplifies the process. Since the first player can choose to be on either side—**{CC, CD}** or **{DC, DD}**—the space of long-run frequencies is constrained to a two-dimensional subspace defined solely by the first player. Because the payoff is linear in these frequencies, the decision boundary for Nash equilibria becomes clear: a strategy can only be Nash if its self-payoff lies at an extreme of this linear function. Once a game {R, S, T, P} is set, we can adjust the strategy vector and visually determine if it is Nash!

<iframe src="/simplex.html" width="800" height="900" frameborder="0"></iframe>

This idea can be formulated analytically, yielding simple decision boundaries. For the full classification of all Nash Equilibria, see [WIP].
