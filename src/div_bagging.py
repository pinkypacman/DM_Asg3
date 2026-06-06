import numpy as np, time
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

C = "cache/"; NC = 6; NJ = 6
B = np.array([0.332, 0.06, 0.991, 0.289, -0.077, -0.066])  # v5 frozen biases

d = np.load(C + "features_richest.npz", allow_pickle=True)
F_train = np.nan_to_num(d["F_train"].astype(np.float64))
F_test = np.nan_to_num(d["F_test"].astype(np.float64))
y_train = d["y_train"].astype(np.int64)
file_id_test = d["file_id_test"].astype(np.int64)
fold_of = np.load(C + "folds.npz")["fold_of"]
y = np.load(C + "gbdt_richest.npz")["y_oof"]

def L(fn, k): return np.load(C + fn, allow_pickle=True)[k].astype(float)
v18 = (0.30 * L("postprocess_tcn_la05.npz", "oof_raw") + 0.23 * L("gbdt_richest.npz", "oof")
       + 0.24 * L("xgb_richest.npz", "oof") + 0.23 * L("_catboost_oof.npz", "oof"))
v18 /= v18.sum(1, keepdims=True)

def postbias_macro(p):
    lp = np.log(np.clip(p, 1e-8, 1)) + B
    return f1_score(y, lp.argmax(1), average="macro", labels=range(NC), zero_division=0)
V18_BASE = postbias_macro(v18)

def run_bagging(class_weight):
    oof = np.zeros((len(y_train), NC)); test = np.zeros((len(F_test), NC))
    for f in range(5):
        tr = fold_of != f; va = fold_of == f
        et = ExtraTreesClassifier(n_estimators=800, max_features="sqrt", min_samples_leaf=2,
                                  class_weight=class_weight, n_jobs=NJ, random_state=42)
        rf = RandomForestClassifier(n_estimators=800, max_features="sqrt", min_samples_leaf=2,
                                    class_weight=class_weight, n_jobs=NJ, random_state=42)
        et.fit(F_train[tr], y_train[tr]); rf.fit(F_train[tr], y_train[tr])
        oof[va] = 0.5 * (et.predict_proba(F_train[va]) + rf.predict_proba(F_train[va]))
        test += 0.5 * (et.predict_proba(F_test) + rf.predict_proba(F_test)) / 5.0
    return oof, test

results = {}
for cw_name, cw in [("balanced_subsample", "balanced_subsample"), ("balanced", "balanced"), ("none", None)]:
    t0 = time.time()
    oof, test = run_bagging(cw)
    oof_n = oof / oof.sum(1, keepdims=True)
    standalone = postbias_macro(oof_n)
    disagreement = float((v18.argmax(1) != oof_n.argmax(1)).mean() * 100)
    mixed = 0.85 * v18 + 0.15 * oof_n; mixed /= mixed.sum(1, keepdims=True)
    blend_lift = postbias_macro(mixed) - V18_BASE
    results[cw_name] = (oof, test, standalone, disagreement, blend_lift)
    print(f"[{cw_name}] standalone={standalone:.4f} disagree={disagreement:.2f}% blend_lift={blend_lift:+.4f} ({time.time()-t0:.0f}s)", flush=True)

# Pick the class_weight maximizing blend_lift (the honest objective)
best = max(results, key=lambda k: results[k][4])
oof, test, standalone, disagreement, blend_lift = results[best]
print(f"\nBEST class_weight={best}")
print(f"V18_BASE={V18_BASE:.4f}")
print(f"standalone_oof_macro={standalone:.4f}")
print(f"raw_oof_macro={f1_score(y, (oof/oof.sum(1,keepdims=True)).argmax(1), average='macro', labels=range(NC), zero_division=0):.4f}")
print(f"disagreement_pct={disagreement:.2f}")
print(f"blend_lift={blend_lift:+.4f}")
# per-class F1 standalone post-bias
lp = np.log(np.clip(oof/oof.sum(1,keepdims=True), 1e-8, 1)) + B
print("per_class_F1_standalone:", np.round(f1_score(y, lp.argmax(1), average=None, labels=range(NC), zero_division=0), 3))
# per-class F1 of mixed vs v18
mixed = 0.85*v18 + 0.15*(oof/oof.sum(1,keepdims=True)); mixed/=mixed.sum(1,keepdims=True)
lpm = np.log(np.clip(mixed,1e-8,1))+B; lpv = np.log(np.clip(v18,1e-8,1))+B
print("per_class_F1 v18  :", np.round(f1_score(y, lpv.argmax(1), average=None, labels=range(NC), zero_division=0), 3))
print("per_class_F1 mixed:", np.round(f1_score(y, lpm.argmax(1), average=None, labels=range(NC), zero_division=0), 3))

oof_save = (oof / oof.sum(1, keepdims=True)).astype("float32")
test_save = (test / test.sum(1, keepdims=True)).astype("float32")
np.savez_compressed(C + "div_bagging.npz", oof=oof_save, test=test_save,
                    y_oof=y_train.astype("int64"), file_id_test=file_id_test)
print("SAVED cache/div_bagging.npz")
