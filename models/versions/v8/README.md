# v8 Multi-Symbol Next-Day OHLC Model

This version extends v7 to support multiple NSE symbols with identical data availability: RELIANCE, TCS, NTPC, and TATAMOTORS. It predicts next-day OHLC for intraday strategy use.

Key points:
- Same architecture and features as v7 (LSTM + multi-task heads).
- Parameterized by `--symbol` to train and run per instrument.
- Per-symbol checkpoints/artifacts/predictions named with the symbol (e.g., `v8_lstm_ohlc_TCS_*`).

Data assumptions:
- Daily: `data/training_data/{SYMBOL}/daily/{SYMBOL}_NSE_2017-10-01_to_2024-12-31.csv`
- 5min:  `data/training_data/{SYMBOL}/5min/{SYMBOL}_NSE_2017-10-01_to_2024-12-31.csv`

Usage:
- Train one symbol:
  - python models/versions/v8/scripts/train_v8.py --symbol TCS
- Train all symbols listed in config:
  - python models/versions/v8/scripts/train_all_v8.py
- Predict for one symbol:
  - python models/versions/v8/scripts/predict_v8.py --symbol TCS

Artifacts:
- Checkpoints: `models/versions/v8/checkpoints/v8_lstm_ohlc_{SYMBOL}_foldK.pt`
- Scalers and temps: `models/versions/v8/artifacts/v8_lstm_ohlc_{SYMBOL}_...`
- Latest prediction JSON: `data/predictions/v8_lstm_ohlc_{SYMBOL}_latest.json`

Notes:
- Walk-forward split, loss weights, calibration, and postprocess options are controlled via `config.yaml`.
- You can adjust `symbols` list in `config.yaml` to add/remove instruments that follow the same folder layout.
