# Causal inference

<img width="459" height="185" alt="xkcd_correlation" src="https://github.com/user-attachments/assets/ca8be03c-031e-4642-a73f-9eecc50386a7" />

*Source: [xkcd](https://xkcd.com/552/)*


### Rubin causal model

- $\mathrm{Y}(0)$: Outcome if the treatment is not administered to the subject.
- $\mathrm{Y}(1)$: Outcome if the treatment is administered to the subject.
- Only one of these outcomes can be known, because the counterfactual outcome is never observed.
- Individual treatment effect for the $i$-th subject: $\mathrm{Y}_i(1) - \mathrm{Y}_i(0)$
- Average treatment effect (ATE): $\tau := \mathbb{E}\big[Y(1) - Y(0)\big] = \mathbb{E}\big[Y(1)\big] - \mathbb{E}\big[Y(0)\big]$

### Acronyms

- ATE: Average Treatment Effect
- CATE: Conditional Average Treatment Effect
- DAG: Directed Acyclic Graph
- DiD: Difference in Differences
- EHRs: Electronic Health Records
- RCTs: Randomized Controlled Trials

### Bibliography

- I Bica, AM Alaa, C Lambert, and M van der Schaar (2021). [**"From Real-World Patient Data to Individualized Treatment Effects Using Machine Learning: Current and Future Methods to Address Underlying Challenges."**](https://pubmed.ncbi.nlm.nih.gov/32449163/) Clinical Pharmacology & Therapeutics.
