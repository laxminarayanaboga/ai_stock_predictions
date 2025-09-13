That’s already a solid foundation — 7+ years of daily data and 26 features (including technical indicators) is good. But if your model still struggles, you can improve it in a few directions:

---

## 🔹 Data Improvements

1. **Higher Frequency Data**

   * Daily candles smooth out a lot of intraday action.
   * Try training with **hourly or 10-min candles** to capture more price dynamics.
   * Then you can aggregate into daily if needed.

2. **Additional Features**

   * **Volatility measures**: ATR, rolling std.
   * **Volume-based features**: OBV, VWAP deviations.
   * **Market-wide data**: index levels (NIFTY, S\&P500), sector indices, VIX.
   * **Calendar features**: day of week, month, expiry day, earnings day.

3. **Regime Detection**

   * Markets behave differently in bull, bear, and sideways phases.
   * Add a “market regime” label (e.g., based on 200-day MA slope or volatility regime) as an input.

4. **Feature Engineering**

   * Instead of just raw RSI/MACD, use **relative features** (e.g., RSI vs its rolling mean).
   * Ratios like High/Low, Close/Open, today’s range vs ATR.

---

## 🔹 Model Improvements

5. **Hybrid Approach**

   * Instead of predicting OHLC directly, predict **direction (up/down)** + **volatility/range** separately.
   * Example: classifier for bullish/bearish, regressor for range size.

6. **Sequence Models**

   * LSTMs / Transformers can model time dependencies better than plain MLP/XGBoost.
   * Sliding window of last N days as input → predict next day OHLC.

7. **Error Distribution Awareness**

   * Don’t just minimize MSE.
   * Train model to predict **uncertainty / confidence interval** (e.g., quantile regression).
   * Helps in risk management — you trade only when model is confident.

---

## 🔹 Strategy-Oriented Training

8. **Direct PnL Optimization**

   * Instead of predicting OHLC, train a model to predict **whether a strategy will be profitable tomorrow**.
   * E.g., label = +1 if long at open & exit at close would give profit, -1 if not.
   * This aligns training with real use.

9. **Ensemble Models**

   * Train multiple models (different feature sets, different lookbacks).
   * Combine them for a stronger signal (majority vote or weighted average).

10. **Walk-Forward Validation**

* Always validate on rolling windows to avoid lookahead bias.
* Markets evolve — retrain periodically (every 3–6 months).

---

👉 Big picture: Right now you’ve trained a **price predictor**. You might get better real-world performance by training a **trading signal predictor** (direct classification of up/down, or trade/no-trade), with features crafted for trading decisions.
