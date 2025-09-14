# v9 - LSTM with Attention and Training Tweaks

What’s new vs v8
- Attention pooling over LSTM outputs for better use of the lookback window
- Early stopping and optional cosine LR scheduler
- Optional class-weighted direction loss
- Same data pipeline and targets as v8 to keep compatibility

Train
- Single symbol:
  python models/versions/v9/scripts/train_v9.py --symbol RELIANCE
- Quick smoke (fewer epochs/folds):
  python models/versions/v9/scripts/train_v9.py --symbol TCS --epochs 10 --max-folds 2

Predict
- After training, generate an ensembled prediction JSON across available folds:
  python models/versions/v9/scripts/predict_v9.py --symbol RELIANCE

Artifacts
- Checkpoints: models/versions/v9/checkpoints
- Scalers/temps: models/versions/v9/artifacts
- Predictions: data/predictions

Notes
- Keep v8 intact; v9 is side-by-side so you can switch back anytime.
- If TA-Lib is unavailable, the pipeline falls back to pure-Pandas indicators.