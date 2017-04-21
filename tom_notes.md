# Experiment 1: gridworld
Variant of "emergence of grounded compo language..."

We choose a gridworld instead of a continuous state space so that both Q and RPG are comparable.

x: instead of being pos, velocity and gaze: only position.

At each timestep, the observation of agent i is:

$o_i(t) = [i, x_i, c_1..N, m_i, g_i]$

Where:

- $x_i(t)$: positions of all the other agents, positions of all the other landmarks at timestep t
- $c_j(t)$: token emitted by agent j at timestep t
- $m_i(t)$: memory of agent i
- $g_i(t)$: goal vector

Landmark positions and goal vectors change across episodes only, the other quantities at each timestep.

Observable: 

