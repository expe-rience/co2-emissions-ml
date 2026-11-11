"""SYNTHETIC data with the same schema as CO2_Emissions.csv. For smoke tests ONLY - not real vehicles.
Usage: python scripts/make_synthetic_data.py out.csv [--drift]   (--drift simulates a changed fleet/regulation)"""
import sys, numpy as np, pandas as pd
def make(n=4000, seed=0, drift=False):
    r = np.random.default_rng(seed)
    vc = r.choice(["COMPACT","SUV - SMALL","MID-SIZE","PICKUP TRUCK - STANDARD","SUBCOMPACT","FULL-SIZE"], n)
    tr = r.choice(["A6","AS8","M6","AM7","AV"], n); ft = r.choice(["X","Z","D","E"], n, p=[.4,.45,.1,.05])
    cyl = r.choice([3,4,6,8,10,12], n, p=[.03,.5,.3,.14,.02,.01]); eng = np.clip(cyl*0.5+r.normal(0.4,.4,n),1.0,8)
    cons = 3.5 + 1.2*eng + 0.5*(ft=="E")*3 + (ft=="D")*(-0.5) + r.normal(0,.7,n) + (vc=="PICKUP TRUCK - STANDARD")*2
    if drift: cons = cons*1.15 + (ft=="D")*2.5; eng = eng*1.1   # fleet/engine behaviour shift
    co2 = 22.5*cons + (ft=="D")*(18 if not drift else 40) - (ft=="E")*40 + r.normal(0,5,n)
    return pd.DataFrame({"Vehicle Class":vc,"Engine Size(L)":eng.round(1),"Cylinders":cyl,"Transmission":tr,
        "Fuel Type":ft,"Fuel Consumption Comb (L/100 km)":cons.round(1),"CO2 Emissions(g/km)":co2.round(0)})
if __name__ == "__main__":
    make(drift="--drift" in sys.argv, seed=7 if "--drift" in sys.argv else 0).to_csv(sys.argv[1], index=False)
