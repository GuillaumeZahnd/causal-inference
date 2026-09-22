# Causal world model

A world model uses a data-driven, model-based approach to capture unknown system dynamics. Once trained, it provides a fast, risk-free offline environment for counterfactual evaluation, answering the core question: *"If I force this action under that state, what happens next?"*

## Paradigm shift

**Predictive model (policy focus):** Given a state, directly map to an action that maximizes expected reward.

*"What action should I take?"*

$$\pi(a \mid s) \rightarrow a^*$$

**Prescriptive model (dynamics focus):** Given the current state and a hypothetical intervention (action), simulate the counterfactual state transition and reward.

*"If I force this action under that state, what happens next?"*

$$M(s_t, a_t) \rightarrow (\hat{s}_{t+1}, \hat{r}_t)$$

## Why use a world model at all?

- We often do not have access to the exact underlying mechanism, instead we can only observe samples that are rather limited in quantity, quality, and diversity, from logged operational data.
- A learned world model is often much faster to run than a physics simulator.
- A neural world model is differentiable by construction, a prerequisite for direct policy optimization.

## Howto

1. Run an agent on simulated data from the environment to generate the consequences of a variety of actions, thereby constructing a plausible dataset from a supposedly unknown underlying mechanism.

```sh
uv run plot_agent_trajectory.py
``

2. Train the world model

Train (and evaluate) the world model, using an agent taking decisions from simulated data from the environment.

```sh
uv run train_world_model.py
``
