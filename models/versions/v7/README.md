# v7 Next-Day OHLC Model

This version predicts next-day OHLC for RELIANCE (NSE) for intraday strategy use.

Key ideas:
- Multi-task: predict next_open (abs) and relative deltas for H/L/C.
- Sequence model (LSTM) over engineered daily features with previous-day intraday aggregates.
- Walk-forward validation and periodic retraining.

See `config.yaml` for settings. Scripts in `scripts/`:
- `train_v7.py` — walk-forward training, checkpoints, metrics
- `predict_v7.py` — daily inference using latest checkpoint
