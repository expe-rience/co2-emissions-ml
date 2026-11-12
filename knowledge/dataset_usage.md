# How this project uses each dataset column

The model uses Vehicle Class, Engine Size(L), Cylinders, Transmission and Fuel Type in the spec_only scenario. The spec_plus_consumption scenario adds Fuel Consumption Comb (L/100 km). The target is CO2 Emissions(g/km).

## Columns that are not used

Make and Model are not used. They have many distinct values, which would create hundreds of one-hot columns and let the model memorise individual cars instead of learning general patterns.

Fuel Consumption City (L/100 km) and Fuel Consumption Hwy (L/100 km) are not used because the combined value already summarises them and they are highly correlated with it.

Fuel Consumption Comb (mpg) is not used because it is the same measurement as Comb (L/100 km) in different units, so adding it would only duplicate information.

## Why fuel consumption is kept as a separate scenario

Fuel consumption is a near-direct predictor of CO2, but it is only known once a vehicle has been fuel-tested. The spec_only scenario exists for vehicles that have not been tested yet.
