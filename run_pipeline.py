"""run_pipeline.py -- one command to rebuild everything:  python run_pipeline.py [--skip-compare] [--tune]"""
import argparse
import importlib

ap = argparse.ArgumentParser()
ap.add_argument("--skip-compare", action="store_true", help="skip the (slower) 14-model comparison")
ap.add_argument("--tune", action="store_true", help="run 4_tune.py first (uses tuned params only if you keep best_params.json)")
a = ap.parse_args()

steps = [("preprocess", "Preprocess"), ("2_train", "Train"), ("3_evaluate", "Evaluate")]
if a.tune:
    steps.insert(1, ("4_tune", "Tune"))
if not a.skip_compare:
    steps.append(("model_comparison", "Compare models"))
for mod, label in steps:
    print(f"\n=== {label} ===")
    importlib.import_module(mod).main()
print("\nDone. Launch the app with:  streamlit run app.py")
