# Causal world model

A world model uses a data-driven, model-based approach to capture unknown system dynamics. Once trained, it provides a fast, risk-free offline environment for counterfactual evaluation, answering the core question: *"If I force this action under that state, what happens next?"*

## Paradigm shift

**Predictive model (policy focus):** Given a state, directly map to an action that maximizes expected reward.

*"What action should I take?"*

$$\pi(a \mid s) \rightarrow a^*$$

**Prescriptive model (dynamics focus):** Given the current state and a hypothetical intervention (action), simulate the counterfactual state transition and reward.

*"If I force this action under that state, what happens next?"*

$$M(s_t, a_t) \rightarrow (\hat{s}_{t+1}, \hat{r}_t)$$
