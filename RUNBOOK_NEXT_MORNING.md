# Next-Morning Runbook (AI Stock Predictions)

Date prepared: 2025-09-17

## What changed today
- Strategy adherence: Only place orders when the generator emits a signal AND `should_trade` is true. Otherwise, strictly NO_TRADE.
- BO TP/SL fix: Distances are now anchored to the actual entry price.
  - BUY: TP = predicted High − entry, SL = entry − predicted Low.
  - SELL: TP = entry − predicted Low, SL = predicted High − entry.
  - Prevents targets exceeding model’s High/Low when actual entry ≠ predicted Open.
- Prediction date correctness during market hours: We drop “today” (IST) rows before feature generation so the model predicts for TODAY (no drift to tomorrow when fresh bars append mid-session).
- Daily updater safety: Skips appending today’s partial 1D candle (IST).
- Wait logging: Clear messaging when wait timed out vs reached target time.
- Dedupe guard: Skips a second order for the same symbol+prediction_date unless `--allow-duplicate` is passed.
- Git hygiene: `logs/` is now ignored. Previously tracked 2025-09-16 logs were removed from the repo.

## How to run tomorrow
Default mode is dry-run. Live trading requires explicit flags and `TRADE_ENABLED`.

- Dry-run (all symbols)
  - `python -m utilities.scripts.one_click_trade`
- Dry-run (subset)
  - `python -m utilities.scripts.one_click_trade --symbols NTPC,TCS,RELIANCE`
- Live (place orders) with confirmation and prefer actual open
  - `TRADE_ENABLED=true python -m utilities.scripts.one_click_trade --no-dry-run --confirm --prefer-actual-open --symbols NTPC,TCS,RELIANCE`
- Wait until open (IST) then act
  - `TRADE_ENABLED=true python -m utilities.scripts.one_click_trade --no-dry-run --confirm --wait-until-open --open-time 09:16`

Notes:
- Set `TRADE_ENABLED=true` in the same command line (the module checks env at import/placement time).
- Use `--allow-duplicate` only if you intentionally want a second order for the same symbol/day.

## Morning checklist
- Fyers session: Script auto-refreshes; look for “Fyers session is valid.”
- Data freshness: Step 2 should show recent 5-min bars (>= 09:15 IST today).
- Prediction date: Ensure printed `prediction_date` equals TODAY.
- Strategy outcome: Only place orders for symbols with signal + `should_trade`.
- Skips: Dedupe prevents re-entry for same symbol/day; use `--symbols` to target specific names.

## Known caveats and small follow-ups
- Dedupe ledger condition: Currently skips when a prior run recorded `decision == ORDER_PLACED`. Our Fyers placement functions don’t return a response, so decision may be `PLACEMENT_ATTEMPTED` instead. Two options:
  1) Make placement functions return the API response and propagate it so `decision` becomes `ORDER_PLACED` on success.
  2) Extend ledger to also consider `PLACEMENT_ATTEMPTED` for the same symbol/date.
- Strict-first-bar mode (optional): Add a mode to wait until the first 5-min candle for TODAY exists in the CSV before generating/deciding, to reduce 09:15/09:16 timing edge cases.
- Active order guard (optional): Query Fyers open orders/positions and skip symbols with an active BO to avoid duplicates across processes.

## Where in code
- Orchestrator: `utilities/scripts/one_click_trade.py`
  - Flags: `--no-dry-run`, `--confirm`, `--prefer-actual-open`, `--wait-until-open`, `--open-time`, `--allow-duplicate`, `--symbols`, `--capital`
  - Dedupe ledger: scans recent `logs/trade_run_*.json` before decisions
  - Wait loop: clearer logs + post-wait data refresh
- Fyers placement: `api/fyers_purchase_management.py`
  - `TRADE_ENABLED` gating; BO helpers; 0.05 tick rounding
- v9 pipeline: `models/versions/v9/data_pipeline.py`
  - Intraday aggregate date alignment (+1 day) is intentional; we filter “today” rows before features to keep prediction on TODAY during market hours.

## Git/logs
- `logs/` is ignored now. New logs won’t be tracked.
- Old 2025-09-16 logs were removed from the repo tip; they remain locally.

---
Quick plan for next improvements (if time permits before open):
- Return placement responses to mark `ORDER_PLACED` correctly and strengthen dedupe.
- Add `--strict-first-bar` flag to ensure the first 5-min bar exists before decisions.
- Optional: Active-order check against Fyers to skip symbols already in a live BO.
