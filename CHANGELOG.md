# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Corrected DST-safe session labels to prevent future pivot leakage and split sessions.
- Made event and Numba/fallback execution consistent, gap-aware, and cadence-independent.
- Corrected drawdown baselines, risk-ratio scaling, non-finite metric handling, and costs.
- Enforced causal ATR/trend features, ordered timestamps, and cadence-aware initial balance.
- Hardened walk-forward, meta-label, multiple-testing, selection, and pivot edge cases.
- Replaced standalone examples with executable public-API examples and expanded release checks.

## [0.1.0] - 2026-05-11

### Added

- Initial release of sessionpt.
- Session-aware OHLC aggregation.
- Six pivot point calculations (classic, woodie, camarilla, DeMark, fibonacci, traditional).
- Event-driven backtest engine with SL/TP exits.
- Vectorized backtest engine with SL/TP grid search.
- Walk-forward fold generator, search helpers, and validation utilities.
- ATR, VWAP, wick, volume, initial-balance, pivot-shift, and confluence helpers.
- Robust analytics for drawdown, Sortino, Calmar, Kelly, and strategy checks.
- Example scripts using synthetic data.
- CI pipeline with ruff, mypy, and pytest.
- PyPI publish workflow on GitHub release.
