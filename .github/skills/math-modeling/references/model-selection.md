# Model Selection Guide

## Match the Problem to the Smallest Useful Model

| Problem pattern | Good first choice | Common fallback |
| --- | --- | --- |
| Allocation, scheduling, routing, assignment | Linear or integer programming | Greedy heuristic plus local search |
| Forecasting a time trend | Regression or time series model | Piecewise regression or exponential smoothing |
| Ranking, scoring, comprehensive evaluation | Entropy weight, TOPSIS, PCA, AHP | Weighted score with sensitivity checks |
| Dynamic population, contagion, diffusion | Difference equations or ODE model | Agent-based or Monte Carlo simulation |
| Queueing, waiting, service capacity | Queueing model or discrete simulation | Empirical rule-based simulation |
| Network influence or path problems | Graph metrics and shortest path models | Network flow or centrality analysis |
| Classification of states or risk levels | Logistic regression or tree model | Rule-based threshold system |
| Multi-factor explanatory analysis | Multiple regression or factor analysis | Correlation analysis with interpretable indices |

## Quick Filters

- If the task says best, least, shortest, minimum, assign, or schedule, start from optimization.
- If the task says predict, estimate next, trend, or future, start from regression or time series.
- If the task says evaluate, rank, compare, or score, start from a multi-criteria evaluation model.
- If the task says spread, growth, interaction, evolution, or stages, start from dynamic models.
- If the task says path, network, transfer, or connection, start from graph models.

## Competition Heuristics

- Prefer interpretable models before complex models.
- Add complexity only when it improves fit, stability, or decision usefulness.
- Keep one fallback model ready in case the data quality is weaker than expected.
- Reserve time for robustness checks and paper cleanup.