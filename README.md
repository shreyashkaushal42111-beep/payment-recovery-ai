# Smart Payment Retry Predictor (AI Revenue Recovery)

**Track 3: AI Revenue Recovery — Razorpay AI Builder Internship 2026**

## Problem it solves
When an online payment fails (network error, OTP timeout, insufficient
funds, etc.), businesses lose revenue if they don't retry at the right
time or in the right way. Blindly retrying every failed transaction
wastes gateway calls and annoys customers, while ignoring recoverable
failures loses real revenue.

This project builds an ML model that:
1. Predicts the **probability that retrying a failed payment will succeed**,
   based on the failure reason, payment method, amount, time of day, and
   customer history.
2. Recommends the **best hour to retry** for each type of failure.
3. Suggests an action: auto-retry now, retry later, or send an alternate
   payment link/reminder instead of retrying blindly.

## Project Objective (for the form)
> Reduce revenue loss from failed payments by using a machine learning
> model to predict retry-success probability and recommend the optimal
> retry timing/strategy per transaction, instead of retrying all failed
> payments uniformly.

## How it works
- `src/generate_data.py` — creates a synthetic dataset of 3,000 failed
  transactions (simulating a real payment gateway's failure logs).
- `src/train_model.py` — trains a Random Forest classifier to predict
  retry success, and computes the best retry hour per failure reason.
- `src/predict.py` — demo script: feed in a new failed transaction, get
  a success probability + recommended action.

## Results
- Model ROC-AUC: **0.79**
- Failure reason and transaction amount are the strongest predictors of
  retry success.
- Example insight: `bank_server_error` and `network_timeout` failures
  recover well on retry (~85%+ success) if retried at low-traffic hours
  (~12–1 AM), while `card_expired` failures should NOT be auto-retried
  (should route to "update payment method" flow instead).

## Tech stack
Python, pandas, scikit-learn (Random Forest), joblib

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py     # generates data/failed_transactions.csv
python src/train_model.py       # trains model, saves retry_model.pkl
python src/predict.py           # demo prediction on a sample transaction
```

## Future scope
- Connect to a real payment gateway's transaction webhook.
- Add a dashboard (Streamlit) to visualize recovery rate over time.
- A/B test retry-timing strategy vs. current uniform-retry approach.
