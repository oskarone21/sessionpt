# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
