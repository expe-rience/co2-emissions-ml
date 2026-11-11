import argparse, json
import pandas as pd
from .config import load_config
from .data import load_data, split, validate
from .train import run_training
from .monitor import monitor_and_retrain, check
from .registry import Registry

def main(argv=None):
    p = argparse.ArgumentParser("co2-ml", description="CO2 emissions model: train / predict / monitor / retrain")
    p.add_argument("--config", default="configs/config.yaml")
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train", help="compare all models, tune top-K, register champion"); t.add_argument("--data")
    pr = sp.add_parser("predict", help="batch predict from CSV"); pr.add_argument("--input", required=True)
    pr.add_argument("--output", default="predictions.csv"); pr.add_argument("--scenario", default="spec_plus_consumption")
    for n, h in [("monitor", "check champion on new labeled data (no retrain)"),
                 ("retrain", "check and auto tune/retrain/promote if metrics dropped")]:
        m = sp.add_parser(n, help=h); m.add_argument("--new-data", required=True)
        m.add_argument("--reference-data", help="original training CSV (for drift + retrain pool)")
        m.add_argument("--scenario", default="spec_plus_consumption")
    a1 = sp.add_parser("ask", help="ask the RAG assistant"); a1.add_argument("question")
    sp.add_parser("index", help="(re)build the RAG vector index")
    sv = sp.add_parser("serve", help="run the HTTP API"); sv.add_argument("--port", type=int, default=8000)
    sp.add_parser("history").add_argument("--scenario", default="spec_plus_consumption")
    a = p.parse_args(argv); cfg = load_config(a.config)

    if a.cmd == "train":
        r = run_training(cfg, a.data)
        for s, v in r.items(): print(f"\n[{s}] {v['version']} {v['model']} {v['metrics']}")
    elif a.cmd == "predict":
        model, meta = Registry(cfg["registry_dir"]).load_champion(a.scenario)
        df = pd.read_csv(a.input); df["Predicted CO2 Emissions(g/km)"] = model.predict(df[meta["features"]])
        df.to_csv(a.output, index=False); print(f"wrote {a.output} using {meta['model']} {meta['version']}")
    elif a.cmd in ("monitor", "retrain"):
        new = load_data(cfg, a.new_data); validate(cfg, new)
        ref = load_data(cfg, a.reference_data) if a.reference_data else None
        res = monitor_and_retrain(cfg, a.scenario, new, ref, auto_retrain=(a.cmd == "retrain"))
        print(json.dumps(res, indent=2, default=float))
    elif a.cmd == "ask":
        from .service import Service; r = Service(a.config).ask(a.question)
        print(r["answer"]); [print(f"  [{x['n']}] {x['source']} (score {x['score']})") for x in r["sources"]]
    elif a.cmd == "index":
        from .rag import RAGPipeline; r = RAGPipeline(cfg); r.reindex(); print(f"indexed {len(r.index.chunks)} chunks")
    elif a.cmd == "serve":
        import os, uvicorn; os.environ["CO2_CONFIG"] = a.config; uvicorn.run("co2_ml.api:app", host="0.0.0.0", port=a.port)
    elif a.cmd == "history":
        for h in Registry(cfg["registry_dir"]).history(a.scenario):
            print(h["version"], h["created"], h["model"], round(h["metrics"]["rmse"], 3), h["trigger"])
if __name__ == "__main__":
    main()
