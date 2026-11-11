# How the assistant works and its limits

## What the assistant does
The assistant answers questions about the CO2 model by retrieving relevant passages from this knowledge base, the model registry and the monitoring reports, then writing an answer that cites those passages. When a vehicle is supplied it also receives the live prediction and the SHAP explanation for that vehicle.

## What it does not do
The assistant does not produce the prediction. The prediction always comes from the trained regression model. The language layer only explains and summarises. If the retrieved sources do not contain the answer, the assistant says so instead of guessing.

## Known limitations of the model
Predictions are estimates for light-duty vehicles in the training data. Vehicles far outside the training range, such as unusual fuel types or very large engines, are extrapolations and should be treated with caution. Estimates are tailpipe CO2 in grams per km, not lifecycle emissions.
