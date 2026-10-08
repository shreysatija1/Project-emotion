import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config import *
from cache import load_features, load_split

tr, va, te, y = load_split()
RESULTS_DIR.mkdir(exist_ok=True)
all_rows = []
for kind in ["mfcc", "mfmc"]:
    rows = []
    for n in COEFF_LIST:
        X = load_features(kind, n)
        sc = StandardScaler().fit(X[tr])          # scaler fitted on TRAIN only
        Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
        models = [("SVM", SVC(kernel="rbf", C=10, gamma="scale")),
                  ("LogReg", LogisticRegression(max_iter=3000))]
        for name, clf in models:
            clf.fit(Xtr, y[tr])
            rows.append({"features": kind.upper(), "n_coeff": n, "vector_size": X.shape[1],
                         "classifier": name,
                         "val_acc": round(accuracy_score(y[va], clf.predict(Xva)), 4),
                         "test_acc": round(accuracy_score(y[te], clf.predict(Xte)), 4)})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"sanity_{kind}.csv", index=False)
    all_rows += rows
comp = pd.DataFrame(all_rows)
comp.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
print(comp.to_string(index=False))
print("Majority-class baseline (always guess anger): about", round(127 / 535, 3))
