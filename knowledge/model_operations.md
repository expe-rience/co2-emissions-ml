# Model operations runbook

## How the champion model is chosen
The training command compares thirteen regression algorithms with five-fold cross-validation and a held-out test split. The best three are tuned with randomized search. The model with the lowest test RMSE becomes the champion and is stored in the registry with its metrics, parameters and a background sample used for explanations.

## Degradation triggers
The monitor scores the champion on new labelled data. A model is flagged as degraded when RMSE rises more than 10 percent over its baseline, MAE rises more than 10 percent, or R-squared falls by more than 0.02. These limits are set in configs/config.yaml under monitoring.

## Drift
Drift is measured with the Population Stability Index (PSI) between the training data and new data. PSI below 0.1 means stable, 0.1 to 0.25 means moderate shift, and above 0.25 means significant drift and should trigger investigation. Drift can be detected without labels, so it is an early warning before accuracy visibly drops.

## Retraining and promotion
When degradation is detected the retrain command re-tunes candidate models on the reference data plus recent data. The new model replaces the champion only if it beats the old champion by at least 1 percent RMSE on a hold-out split of the new data. Otherwise the old champion stays in production and the new version is stored but not promoted.

## What to do when PSI is high
First check whether the input data pipeline changed, for example units, missing values or a new category. If the data is correct, the real world has changed, for example new vehicle models or fuel types, and retraining on recent data is the correct response. Never promote a retrained model without comparing it with the champion on the same hold-out data.

## Rollback
Every version is kept under models/<scenario>/. To roll back, write the previous version number into models/<scenario>/champion.txt.
