"""Train the Stage-1 suitability surrogate (LightGBM, with XGBoost for comparison).

Trains on >=20k meal instances derived from Food.com meals crossed with sampled
profiles, cross-validates, and writes models to ``models/`` alongside a metrics
JSON reporting RMSE, R^2 and thresholded AUC on held-out Food.com *and* on
Nutrition5k.

TODO(Phase 2): implement.
"""
