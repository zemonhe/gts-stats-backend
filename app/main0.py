
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, io, requests, pandas as pd
from datetime import datetime
from typing import Optional, List
from features import to_result_label, build_match_features
from trainer import build_model, evaluate, backtest, save_model, load_model

load_dotenv()
API_TOKEN = os.getenv("FOOTBALL_DATA_API_TOKEN")
BASE = "https://api.football-data.org/v4"

app = FastAPI(title="GTS STATS Backend", version="1.0.0")
origins = ["http://localhost:3000","http://127.0.0.1:3000","https://*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATA_DIR = os.path.join(os.path.dirname(__file__),"data"); os.makedirs(DATA_DIR, exist_ok=True)
OUT_DIR  = os.path.join(os.path.dirname(__file__),"outputs"); os.makedirs(OUT_DIR, exist_ok=True)
MOD_DIR  = os.path.join(os.path.dirname(__file__),"models_store"); os.makedirs(MOD_DIR, exist_ok=True)

class FetchBody(BaseModel):
    seasons: List[int] = [2022, 2023, 2024]
    leagues: List[str] = ["PL","PD","PORTUGAL_DYNAMIC"]

class TrainBody(BaseModel):
    cutoff: Optional[str] = None

class EvaluateBody(BaseModel):
    cutoff: Optional[str] = None

class BacktestBody(BaseModel):
    splits: int = 5

class PredictBody(BaseModel):
    fixturesCsvPath: str = "data/fixtures.csv"

def now_iso(): return datetime.utcnow().isoformat()
def _headers():
    if not API_TOKEN: raise RuntimeError("FOOTBALL_DATA_API_TOKEN não definido.")
    return {"X-Auth-Token": API_TOKEN}

def discover_portugal_code():
    r = requests.get(f"{BASE}/competitions", headers=_headers(), timeout=30)
    r.raise_for_status()
    comps = r.json().get("competitions", [])
    cand = [c for c in comps if c.get("area",{}).get("name")=="Portugal" and c.get("type")=="LEAGUE" and ("Liga" in c.get("name","") or "Primeira" in c.get("name",""))]
    if not cand: raise RuntimeError("Não foi possível detetar a Liga Portugal.")
    c = cand[0]
    return c.get("code") or c.get("id")

def fetch_comp_matches(comp_code, season, status):
    url = f"{BASE}/competitions/{comp_code}/matches"
    params = {"season": season, "status": status}  # FINISHED | SCHEDULED
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("matches", [])

def matches_to_rows(matches):
    rows=[]
    for m in matches:
        score = (m.get("score") or {}).get("fullTime") or {}
        rows.append({
            "date": pd.to_datetime(m.get("utcDate"), utc=True, errors="coerce"),
            "league": (m.get("competition") or {}).get("name"),
            "season": pd.to_datetime(((m.get("season") or {}).get("startDate")), errors="coerce").year if (m.get("season") or {}).get("startDate") else None,
            "home_team": (m.get("homeTeam") or {}).get("name"),
            "away_team": (m.get("awayTeam") or {}).get("name"),
            "home_goals": score.get("home"),
            "away_goals": score.get("away")
        })
    return rows

@app.get("/health")
def health(): return {"status":"ok","time":now_iso()}

@app.post("/fetch")
def fetch(body: FetchBody):
    leagues = body.leagues.copy()
    if "PORTUGAL_DYNAMIC" in leagues:
        leagues.remove("PORTUGAL_DYNAMIC")
        leagues.append(discover_portugal_code())
    all_rows=[]
    for lg in leagues:
        for season in body.seasons:
            fin = fetch_comp_matches(lg, season, "FINISHED")
            all_rows += matches_to_rows(fin)
    df = pd.DataFrame(all_rows).dropna(subset=["date","home_team","away_team"])
    df["season"] = df["season"].fillna(df["date"].dt.year).astype(int)
    df["result"] = df.apply(lambda r: to_result_label(r["home_goals"], r["away_goals"]), axis=1)
    df = df[df["result"].isin(["H","D","A"])].drop_duplicates(subset=["date","home_team","away_team"]).sort_values("date")
    path = os.path.join(DATA_DIR,"matches_clean.csv")
    df.to_csv(path, index=False)
    return {"rows": int(df.shape[0]), "path": "data/matches_clean.csv", "updatedAt": now_iso()}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    req = {"date","league","season","home_team","away_team","home_goals","away_goals"}
    if not req.issubset(df.columns):
        return {"ok":False, "message": f"Faltam colunas: {list(req - set(df.columns))}"}
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["result"] = df.apply(lambda r: to_result_label(r["home_goals"], r["away_goals"]), axis=1)
    df.to_csv(os.path.join(DATA_DIR,"matches_clean.csv"), index=False)
    return {"ok":True, "rows": len(df), "updatedAt": now_iso()}

@app.post("/train")
def train(body: TrainBody):
    df = pd.read_csv(os.path.join(DATA_DIR,"matches_clean.csv"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    feat = build_match_features(df)
    cutoff = pd.to_datetime(body.cutoff) if body.cutoff else feat["date"].quantile(0.8)
    train_df = feat[feat["date"] < cutoff]; test_df = feat[feat["date"] >= cutoff]
    Xtr = train_df.drop(columns=["date","result"]); ytr = train_df["result"]
    Xte = test_df.drop(columns=["date","result"]); yte = test_df["result"]
    model = build_model().fit(Xtr, ytr)
    met = evaluate(model, Xte, yte)
    save_model(model, os.path.join(MOD_DIR,"model.joblib"))
    return {"cutoff": str(cutoff.date()), "trainedAt": now_iso(), **met}

@app.post("/evaluate")
def evaluate_holdout(body: EvaluateBody):
    df = pd.read_csv(os.path.join(DATA_DIR,"matches_clean.csv"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    feat = build_match_features(df)
    cutoff = pd.to_datetime(body.cutoff) if body.cutoff else feat["date"].quantile(0.8)
    test_df = feat[feat["date"] >= cutoff]
    bundle = load_model(os.path.join(MOD_DIR,"model.joblib"))
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    met = evaluate(model, test_df.drop(columns=["date","result"]), test_df["result"])
    return {"cutoff": str(cutoff.date()), **met}

@app.post("/backtest")
def backtest_api(b: BacktestBody):
    df = pd.read_csv(os.path.join(DATA_DIR,"matches_clean.csv"))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    feat = build_match_features(df)
    res = backtest(feat)
    return res

@app.post("/predict")
def predict(b: PredictBody):
    path = os.path.join(os.path.dirname(__file__), b.fixturesCsvPath)
    if not os.path.exists(path): return {"error":"fixtures.csv não encontrado"}
    fx = pd.read_csv(path)
    hist = pd.read_csv(os.path.join(DATA_DIR,"matches_clean.csv"))
    hist["result"] = hist["result"].fillna("")
    combined = pd.concat([hist, fx.assign(home_goals=pd.NA, away_goals=pd.NA, result=pd.NA)], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")
    feat = build_match_features(combined.sort_values("date"))
    pred_df = feat[feat["result"].isna()]
    bundle = load_model(os.path.join(MOD_DIR,"model.joblib"))
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    proba = model.predict_proba(pred_df.drop(columns=["date","result"]))
    classes = ["H","D","A"]; import numpy as np
    top = np.array(classes)[np.argmax(proba, axis=1)]
    out = pred_df[["date","league","season","home_team","away_team"]].copy()
    out["prob_home"] = proba[:,0]; out["prob_draw"]=proba[:,1]; out["prob_away"]=proba[:,2]
    out["pred_class"] = top
    out.to_csv(os.path.join(OUT_DIR,"fixtures_predictions.csv"), index=False)

