import os, time
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

C = "cache/"
NC = 6
RNG = 42

t0 = time.time()
feat = np.load(C + "features_richest.npz", allow_pickle=True)
F_train = feat["F_train"].astype(np.float64)
F_test = feat["F_test"].astype(np.float64)
y_train = feat["y_train"].astype(np.int64)
file_id_test = feat["file_id_test"].astype(np.int64)
fold_of = np.load(C + "folds.npz", allow_pickle=True)["fold_of"]

# Clean any non-finite values defensively (some hand features can have nan/inf).
F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
F_test = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0)

n_train = F_train.shape[0]
n_test = F_test.shape[0]
n_folds = int(fold_of.max()) + 1

oof = np.zeros((n_train, NC), dtype=np.float64)
test = np.zeros((n_test, NC), dtype=np.float64)


def make_logreg(C_val):
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            C=C_val, class_weight="balanced", max_iter=3000,
            solver="lbfgs", multi_class="multinomial", n_jobs=6,
            random_state=RNG)),
    ])


def make_nystroem_logreg(gamma_val, C_val):
    # Nystroem RBF approximation -> linear LogisticRegression (fast RBF-SVC surrogate).
    return Pipeline([
        ("sc", StandardScaler()),
        ("ny", Nystroem(kernel="rbf", gamma=gamma_val, n_components=300, random_state=RNG)),
        ("clf", LogisticRegression(
            C=C_val, class_weight="balanced", max_iter=3000,
            solver="lbfgs", multi_class="multinomial", n_jobs=6,
            random_state=RNG)),
    ])


# ------- light per-model hyperparameter selection on OOF macro-F1 -------
# To stay within time budget, tune on a single held-out fold (fold 0), then
# lock the choice and run the full 5-fold OOF protocol.
B = np.array([0.332, 0.06, 0.991, 0.289, -0.077, -0.066])  # v5 frozen biases for selection signal


def quick_oof_macro(model_factory, *args):
    # fold-0 holdout quick eval (post-bias macro on the held-out fold)
    f = 0
    tr = fold_of != f
    va = fold_of == f
    m = model_factory(*args)
    m.fit(F_train[tr], y_train[tr])
    p = m.predict_proba(F_train[va])
    full = np.zeros((NC,))
    lp = np.log(np.clip(p, 1e-8, 1)) + B
    return f1_score(y_train[va], lp.argmax(1), average="macro", labels=range(NC), zero_division=0)


# gamma scale heuristic: features standardized -> dim ~196, rbf gamma ~ 1/n_features baseline
n_feat = F_train.shape[1]
print(f"[setup] n_feat={n_feat}")

best_C_lr, best_lr = None, -1
for Cv in [0.1, 0.3, 1.0, 3.0]:
    s = quick_oof_macro(make_logreg, Cv)
    print(f"[tune logreg] C={Cv} fold0_macro={s:.4f}")
    if s > best_lr:
        best_lr, best_C_lr = s, Cv

best_pair_ny, best_ny = None, -1
gamma_base = 1.0 / n_feat
for gm in [gamma_base * 0.5, gamma_base, gamma_base * 2.0]:
    for Cv in [0.5, 1.0, 3.0]:
        s = quick_oof_macro(make_nystroem_logreg, gm, Cv)
        print(f"[tune nystroem] gamma={gm:.5f} C={Cv} fold0_macro={s:.4f}")
        if s > best_ny:
            best_ny, best_pair_ny = s, (gm, Cv)

print(f"[selected] logreg C={best_C_lr} (fold0={best_lr:.4f}) | nystroem gamma={best_pair_ny[0]:.5f} C={best_pair_ny[1]} (fold0={best_ny:.4f})")

# ------- full 5-fold OOF protocol -------
for f in range(n_folds):
    tr = fold_of != f
    va = fold_of == f

    lr = make_logreg(best_C_lr)
    lr.fit(F_train[tr], y_train[tr])
    p_lr_va = lr.predict_proba(F_train[va])
    p_lr_te = lr.predict_proba(F_test)

    ny = make_nystroem_logreg(best_pair_ny[0], best_pair_ny[1])
    ny.fit(F_train[tr], y_train[tr])
    p_ny_va = ny.predict_proba(F_train[va])
    p_ny_te = ny.predict_proba(F_test)

    # average the two calibrated probability outputs
    oof[va] = 0.5 * (p_lr_va + p_ny_va)
    test += 0.5 * (p_lr_te + p_ny_te)
    print(f"[fold {f}] done  ntr={tr.sum()} nva={va.sum()}  t={time.time()-t0:.1f}s")

test /= n_folds

# normalize rows
oof /= oof.sum(1, keepdims=True)
test /= test.sum(1, keepdims=True)

np.savez_compressed(
    C + "div_linear.npz",
    oof=oof.astype("float32"),
    test=test.astype("float32"),
    y_oof=y_train.astype("int64"),
    file_id_test=file_id_test.astype("int64"),
)
print(f"[saved] {C}div_linear.npz  total t={time.time()-t0:.1f}s")
