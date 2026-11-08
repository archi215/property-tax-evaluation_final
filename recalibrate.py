"""recalibrate.py -- rebuild the conformal intervals (incl. per-property-type) WITHOUT retraining the model.
    python recalibrate.py
Updates models/metadata.json and reports empirical test coverage by property type."""
import json

import numpy as np

import config
import conformal as cf
from common import load_splits
from valuation import ValuationModel


def main():
    vm = ValuationModel()
    _, sp = load_splits()
    (Xc, yc), (Xt, yt) = sp["calib"], sp["test"]
    new = cf.build(vm.model, Xc[vm.features], yc)
    meta = json.load(open(config.METADATA_JSON))
    old_cov = meta.get("coverage_by_type_test")
    meta["conformal_log_quantiles"] = new
    pred = vm.model.predict(Xt[vm.features])
    hit_global = np.abs(yt.values - pred) <= new["global"]["90"]
    cov = {}
    for col in [c for c in Xt.columns if c.startswith("property_type_")]:
        m = Xt[col].values.astype(bool)
        name = col[len("property_type_"):]
        q = cf.pick(new, "", name, "90")
        cov[name] = {"n_test": int(m.sum()), "coverage_90_global_q": float(hit_global[m].mean()),
                     "coverage_90_type_q": float((np.abs(yt.values - pred)[m] <= q).mean()), "n_cal": new["by_type"].get(name, {}).get("n", 0)}
    meta["coverage_by_type_test"] = cov
    json.dump(meta, open(config.METADATA_JSON, "w"), indent=2)
    for k, v in sorted(cov.items()):
        print(f"{k:18s} n_test={v['n_test']:4d} n_cal={v['n_cal']:4d}  coverage global q: {v['coverage_90_global_q']:.1%}  type q: {v['coverage_90_type_q']:.1%}")


if __name__ == "__main__":
    main()
