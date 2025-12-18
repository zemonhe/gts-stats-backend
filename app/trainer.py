
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit

NUM_COLS = [
 "home_ppg_5","home_ppg_10","home_ppg_15","away_ppg_5","away_ppg_10","away_ppg_15",
 "home_gf_5","home_gf_10","home_gf_15","away_gf_5","away_gf_10","away_gf_15",
 "home_ga_5","home_ga_10","home_ga_15","away_ga_5","away_ga_10","away_ga_15",
 "diff_ppg_5","diff_ppg_10","diff_ppg_15","diff_gf_5","diff_gf_10","diff_gf_15","diff_ga_5","diff_ga_10","diff_ga_15"
]
CAT_COLS = ["home_team","away_team","league"]

def build_model():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=300, C=1.0, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return CalibratedClassifierCV(pipe, method="isotonic", cv=3)

def evaluate(model, X, y):
    proba = model.predict_proba(X)
    classes = np.array(["H","D","A"])
    pred = classes[np.argmax(proba, axis=1)]
    acc = accuracy_score(y, pred)
    y_map = {"H":0,"D":1,"A":2}
    y_idx = np.array([y_map[c] for c in y])
    ll = log_loss(y_idx, proba, labels=[0,1,2])
    brier = np.mean([brier_score_loss((y_idx==k).astype(int), proba[:,k]) for k in [0,1,2]])
    cm = confusion_matrix(y, pred, labels=["H","D","A"])
    rep = classification_report(y, pred, labels=["H","D","A"], digits=3)
    return {"accuracy":acc,"log_loss":ll,"brier_score":brier,"confusion_matrix":cm.tolist(),"classification_report":rep}

def backtest(df):
    tss = TimeSeriesSplit(n_splits=5)
    rows=[]
    for i,(tr,te) in enumerate(tss.split(df),1):
        mdl = build_model()
        mdl.fit(df.iloc[tr][NUM_COLS+CAT_COLS], df.iloc[tr]["result"])
        m = evaluate(mdl, df.iloc[te][NUM_COLS+CAT_COLS], df.iloc[te]["result"])
        m.update({"fold":i,"train_end":str(df.iloc[tr]["date"].max()),"test_start":str(df.iloc[te]["date"].min()),"test_end":str(df.iloc[te]["date"].max())})
        rows.append(m)
    return rows

def save_model(model, path): joblib.dump({"model":model,"feature_cols":NUM_COLS+CAT_COLS}, path)
def load_model(path): return joblib.load(path)
