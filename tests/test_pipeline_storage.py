import numpy as np
import pandas as pd
import preprocess, storage


def test_split_fractions_and_disjoint():
    lab = preprocess.assign_splits(1000)
    counts = pd.Series(lab).value_counts()
    assert counts["train"] == 700 and counts["test"] == 100 and counts.sum() == 1000


def test_target_encoding_uses_train_only():
    df = pd.DataFrame({"city": ["A"] * 50 + ["B"] * 50, "p": [1.0] * 50 + [100.0] * 50})
    m, g = preprocess.fit_target_encoding(df.iloc[:60], "city", "p", m=20)
    assert m["A"] < m["B"] and g == df.iloc[:60]["p"].mean()


def test_storage_roundtrip(tmp_path):
    p = str(tmp_path / "h.db")
    rid = storage.save_valuation({"city": "X", "predicted_price": 1.0}, {"a": 1}, path=p)
    assert len(storage.load_history(p)) == 1
    storage.delete_valuation(rid, p)
    assert storage.load_history(p).empty
