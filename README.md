\# ML-Based Long-Only Momentum Strategy



A machine learning equity selection strategy that predicts weekly cross-sectional outperformance across 10 US large-cap stocks using a 4-model ensemble with online adaptation, institutional walk-forward validation, and post-hoc signal diagnostics.



Built for a quantitative finance hackathon. The strategy returned \*\*107.14% cumulative (after costs)\*\* over 157 weeks. The report then proves the returns came from beta exposure, not stock selection skill, and diagnoses exactly why.



\---



\## Results at a Glance



| Metric | Before TC | After TC |

|---|---:|---:|

| Cumulative Return | 127.66% | 107.14% |

| Annualised Return | 31.32% | 27.28% |

| Annualised Volatility | 25.17% | 25.15% |

| Sharpe Ratio | 1.244 | 1.084 |

| Max Drawdown | -26.20% | -26.50% |

| Hit Rate | 56.69% | 56.05% |

| Avg Weekly Return | 0.585% | 0.525% |



Equal-weight buy-and-hold benchmark returned 170.46% over the same period. The strategy underperformed by 63 percentage points. Section 5 of the notebook explains why.



\---



\## Universe and Data



\- \*\*Stocks:\*\* AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, V, JNJ, BRK-B

\- \*\*Data:\*\* Daily OHLCV from Yahoo Finance, resampled to weekly (Friday close)

\- \*\*Train:\*\* 2017-01-01 to 2022-12-31

\- \*\*Test:\*\* 2023-01-01 to 2025-12-31

\- \*\*Fallback:\*\* Correlated multivariate GBM synthetic data if yfinance is unavailable



\---



\## Architecture



\### Target Variable



Binary: did this stock beat the equal-weight market return next week? Subtracting the market return strips beta from the label. The model learns relative outperformance, not market direction.



\### Features (14 per stock-week)



| Group | Count | Description |

|---|---:|---|

| Vol-adjusted momentum | 3 | Return / (vol x sqrt(horizon)) at 1M, 3M, 6M |

| Cross-sectional z-scores | 4 | Per-week z-score of momentum + microstructure across 10 stocks |

| Microstructure proxy | 1 | Weekly avg of (High - Low) / Low (Parkinson range) |

| Lagged returns | 4 | Weekly returns at lags 1 through 4 |

| Regime conditioning | 2 | Portfolio vol z-score + momentum dispersion z-score |



\### Ensemble (4 models, equal-weighted)



1\. \*\*XGBoost\*\* -- Optuna-tuned (60 trials, TPE sampler, IC objective). max\_depth=2, lr=0.030

2\. \*\*LightGBM\*\* -- Mirrored Optuna params, leaf-wise growth

3\. \*\*Logistic Regression\*\* -- C=0.01, balanced classes. Linear baseline

4\. \*\*XGB-Diverse\*\* -- Shallower, heavier regularisation, different seed



Each model outputs probabilities that are rank-normalised weekly, then averaged.



\### Online Adaptation



After each weekly prediction, 3 incremental trees are appended to the primary XGBoost using that week's realised data. Predict first, update after. Zero look-ahead. Regime adaptation half-life of roughly 10 weeks versus 13 weeks with batch-only quarterly retraining.



\### Walk-Forward Validation



\- 13 quarterly out-of-sample folds (expanding training window)

\- 1-week label purge at train/test boundary

\- 2-week feature embargo for autocorrelation buffer

\- Exponential time-decay sample weights (half-life: 104 weeks)



Reference: Lopez de Prado, \*Advances in Financial Machine Learning\* (2018), Ch. 7.



\### Portfolio Construction



\- Select top 2 stocks weekly by ensemble score

\- Equal-weight positions

\- Dynamic inertia rule: held stocks receive +0.50 boost in rank-normalised score space (displaced only when 6+ stocks outrank them)

\- Transaction costs: 0.1% entry + 0.1% exit, charged on changed legs only

\- 43.3% of weeks had zero trades



\---



\## Key Findings



\### The model's signal is inverted for the test period



An inverse-selection backtest (hold the bottom-2 instead of top-2) returned 155% versus the strategy's 107%. The model picks with real structure. The structure is backwards for the post-2022 tech recovery.



\### Root cause: regime break at training cutoff



Training ends December 2022, the year META fell 65% and JNJ fell 4%. The model learned "defensive = good." In 2023 META recovered 400%. The model held JNJ in 47% of weeks and META in under 9%.



\### The architecture works in a different regime



In-sample validation (train 2017-2019, test 2020-2022) returned 112% against a 50% benchmark with positive selection alpha of +0.43%/week and positive Rank IC of +0.015. The failure is regime-specific, not structural.



\### The signal predicts a slower horizon



Rank IC at 1-week: -0.018. At 4-week: +0.039. The model captures a real signal that matures over a month. Weekly rebalancing discards half the signal before it arrives.



\### Statistical robustness



Stationary block bootstrap (5,000 resamples): Sharpe 95% CI of \[-0.05, +2.20]. The interval includes zero. Strategy capacity via Almgren-Chriss square-root impact model: $50-100M before impact erodes alpha.



\---



\## Repository Structure



```

.

├── ml\_momentum\_strategy\_final.ipynb   # Complete end-to-end notebook (49 cells)

├── strategy\_report.docx               # 4-page formal report with embedded figures

├── README.md

└── outputs/

&#x20;   ├── weekly\_predictions.csv          # Model scores for all stocks, all test weeks

&#x20;   ├── portfolio\_log.csv               # Weekly portfolio: selected stocks, trades, returns

&#x20;   ├── performance\_summary.csv         # Before/after TC performance metrics

&#x20;   ├── performance\_dashboard.png       # 6-panel dashboard (cumulative, drawdown, dist, Sharpe, freq, turnover)

&#x20;   ├── ic\_diagnostics.png              # Per-fold IC, weekly IC time series, IC distribution

&#x20;   ├── feature\_importance.png          # XGBoost gain-based top-20 features

&#x20;   ├── selection\_alpha.png             # Selected vs avoided cumulative, edge, scatter

&#x20;   ├── inverse\_selection.png           # Top-2 vs bottom-2 vs benchmark equity curves

&#x20;   ├── insample\_validation.png         # 2020-2022 regime validation results

&#x20;   ├── signal\_decay.png                # IC at 1w, 2w, 3w, 4w horizons

&#x20;   ├── cumulative\_alpha.png            # Cumulative alpha vs benchmark + quarterly breakdown

&#x20;   ├── bootstrap\_ci.png                # Block bootstrap distributions with 95% CIs

&#x20;   └── strategy\_capacity.png           # AUM vs market impact capacity curve

```



\---



\## Notebook Sections



| # | Section | What It Does |

|---|---|---|

| 1 | Imports \& Configuration | Libraries, seeds, output directory |

| 2 | Strategy Configuration | All hyperparameters in one `StrategyConfig` class |

| 3 | Data Acquisition | yfinance download with correlated GBM fallback |

| 4 | Weekly Resampling | OHLCV aggregation to W-FRI convention |

| 5 | Feature Engineering | 14 features, beta-neutral target, long-format stacking |

| 6 | Walk-Forward Engine | Purge + embargo + time-decay + fold generation |

| 7 | Model Training | Optuna tuning, ensemble classifier, online updates, walk-forward runner |

| 8 | Signal Diagnostics | Per-fold Rank IC, weekly IC series, IC distribution |

| 9 | Feature Importance | Gain-based importance, group decomposition |

| 10 | Portfolio Construction | Inertia rule, TC model, weekly rebalancing |

| 11 | Performance Metrics | Sharpe, drawdown, hit rate, before/after TC |

| 12 | Attribution \& Signal Analysis | Selection alpha, ticker-level decomposition, regime conditioning |

| 13 | Performance Dashboard | 6-panel visualisation |

| 14 | Export Deliverables | CSV outputs |

| 15 | Final Scorecard | Summary table, TC drill-down, model diagnostics |

| 16 | Advanced Diagnostics | Inverse backtest, cumulative alpha, signal decay, in-sample validation |

| 17 | Statistical Robustness | Deflated Sharpe Ratio, block bootstrap CIs, strategy capacity |



\---



\## Dependencies



```

numpy

pandas

yfinance

xgboost

lightgbm

scikit-learn

scipy

optuna

matplotlib

seaborn

```



Python 3.10+. No GPU required. Full notebook runs in approximately 4 minutes on a standard laptop (Optuna tuning is the bottleneck).



\---



\## Reproducing Results



```bash

pip install numpy pandas yfinance xgboost lightgbm scikit-learn scipy optuna matplotlib seaborn

jupyter notebook ml\_momentum\_strategy\_final.ipynb

\# Run all cells. Outputs are saved to ./outputs/

```



Results depend on yfinance data availability. If the network is unavailable, the notebook falls back to synthetic GBM data and prints a warning. Synthetic data produces different absolute numbers but exercises the full pipeline.



\---



\## What We Would Change



Five improvements that address diagnosed root causes:



1\. \*\*Mean-reversion feature.\*\* Distance from 26-week rolling high, cross-sectionally z-scored. The 2020 COVID recovery is an in-sample template for crash-recovery dynamics that pure momentum misses.

2\. \*\*Regression target.\*\* XGBRegressor on continuous residual returns. Binary classification throws away magnitude: a stock up 8% and one up 0.01% get the same label.

3\. \*\*Shorter time-decay.\*\* Half-life of 26 weeks instead of 104. At 104 weeks, January 2022 training samples carry 71% weight relative to December 2022. At 26 weeks, that drops to 6%.

4\. \*\*Lower inertia boost.\*\* 0.08 instead of 0.50. The current value locks in wrong selections for quarters. Additional TC drag of roughly 150bp/year is worth paying for regime responsiveness.

5\. \*\*Biweekly rebalance.\*\* Signal decay peaks at 4 weeks. Weekly rebalancing discards half the signal. A 2-week cycle matches the signal's natural timescale and halves turnover.



\---



\## References



\- Lopez de Prado, M. (2018). \*Advances in Financial Machine Learning\*. Wiley. Ch. 7 (Purged Walk-Forward).

\- Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. \*Journal of Portfolio Management\*.

\- Politis, D. \& Romano, J. (1994). The Stationary Bootstrap. \*Journal of the American Statistical Association\*.

\- Almgren, R. \& Chriss, N. (2000). Optimal Execution of Portfolio Transactions. \*Journal of Risk\*.



\---



\## License



This project was built for a hackathon. Use it however you want. If you write about the inverse-selection technique or the signal decay analysis, a mention would be appreciated.

