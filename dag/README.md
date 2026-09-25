### Partially Linear Model (PLM)

$$
\begin{aligned}
Y &= D\theta_0 + g_0(X) + U, \quad &\mathbb{E}[U \mid D, X] &= 0 \\
D &= m_0(X) + V, \quad &\mathbb{E}[V \mid X] &= 0
\end{aligned}
$$

```sh 
               [ X ]
              /     \
             /       \
            v         v
[ V ] ---> [ D ] ----> [ Y ] <--- [ U ]
```

- $Y$: Outcome (e.g., 5-year survival probability)
- $D$: Treatment assignment (e.g., drug dosage or chemotherapy recommendation)
- $X = (X_1, X_2, \dots, X_n)$: Baseline confounding covariates (e.g., age, tumor stage, genetic markers)
- $U$: Exogenous outcome error—unobserved, unconfounded biological variation affecting survival (e.g., unknown post-treatment cellular resilience or unrecorded lifestyle factors)
- $V$: Exogenous treatment innovation—idiosyncratic treatment variation unexplained by patient characteristics (e.g., physician preference, hospital stocking differences, or random trial assignment)
- $\theta_0$: Target causal parameter (Average Treatment Effect of a 1-unit increase in $D$ on $Y$)
- $g_0(X)$: Outcome nuisance function mapping confounders $X \to Y$
- $m_0(X)$: Treatment nuisance function mapping confounders $X \to D$ (propensity score when $D \in \{0, 1\}$)
