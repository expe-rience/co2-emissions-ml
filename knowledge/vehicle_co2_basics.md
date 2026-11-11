# Vehicle CO2 basics

## Why fuel consumption predicts CO2
Burning one litre of fuel releases a nearly fixed amount of CO2. Gasoline releases roughly 2.3 kg CO2 per litre and diesel roughly 2.7 kg per litre, so tailpipe CO2 is almost proportional to fuel burned. A car using 8 L/100 km of gasoline emits about 8 x 10 x 2.3 = 184 g/km. This is why adding the fuel consumption feature lifts R-squared from about 0.86 to 0.99, and why it is a near-direct predictor rather than an independent signal.

## When fuel consumption is not available
If a vehicle has not been tested yet, fuel consumption is unknown, so use the spec_only scenario. It relies on engine size, cylinders, vehicle class, transmission and fuel type, and is less accurate (RMSE around 20 g/km in the original notebook).

## Fuel type codes
X is regular gasoline, Z is premium gasoline, D is diesel, E is ethanol (E85) and N is natural gas. Diesel emits more CO2 per litre but usually uses fewer litres per km. Ethanol vehicles show higher litres per 100 km but lower fossil CO2 per litre.

## Transmission codes
A is automatic, AM is automated manual, AS is automatic with select shift, AV is continuously variable and M is manual. The number after the letters is the count of gears, for example A6 is a 6-speed automatic.

## Engine size and cylinders
Larger engines and more cylinders burn more fuel at the same load, so they usually produce more CO2. The two features are strongly correlated, which is fine for tree models and regularised linear models but can make plain linear coefficients unstable.
