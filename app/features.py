
import numpy as np
import pandas as pd

def to_result_label(hg, ag):
    if pd.isna(hg) or pd.isna(ag): return np.nan
    if hg > ag: return "H"
    if hg < ag: return "A"
    return "D"

def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    base = df[["date","league","season","home_team","away_team","home_goals","away_goals","result"]].copy()
    h = base.copy(); h["team"]=h["home_team"]; h["gf"]=h["home_goals"]; h["ga"]=h["away_goals"]; h["points"]=np.where(h["gf"]>h["ga"],3,np.where(h["gf"]==h["ga"],1,0))
    a = base.copy(); a["team"]=a["away_team"]; a["gf"]=a["away_goals"]; a["ga"]=a["home_goals"]; a["points"]=np.where(a["gf"]>a["ga"],3,np.where(a["gf"]==a["ga"],1,0))
    h = h.sort_values("date"); a = a.sort_values("date")
    for w in [5,10,15]:
        base[f"home_ppg_{w}"] = h.groupby("team")["points"].rolling(w,min_periods=1).mean().shift(1).reset_index(level=0,drop=True)
        base[f"home_gf_{w}"]  = h.groupby("team")["gf"].rolling(w,min_periods=1).mean().shift(1).reset_index(level=0,drop=True)
        base[f"home_ga_{w}"]  = h.groupby("team")["ga"].rolling(w,min_periods=1).mean().shift(1).reset_index(level=0,drop=True)
        base[f"away_ppg_{w}"] = a.groupby("team")["points"].rolling(w,min_periods=1).mean().shift(1).reset_index(level=0,drop=True)
        base[f"away_gf_{w}"]  = a.groupby("team")["gf"].rolling(w,min_periods=1).mean().shift(1).reset_index(level=0,drop=True)
        base[f"away_ga_{w}"]  = a.groupby("team")["ga"].rolling(w,min_periods=1).mean().shift(1).reset_index(level=0,drop=True)
        base[f"diff_ppg_{w}"] = base[f"home_ppg_{w}"] - base[f"away_ppg_{w}"]
        base[f"diff_gf_{w}"]  = base[f"home_gf_{w}"]  - base[f"away_gf_{w}"]
        base[f"diff_ga_{w}"]  = base[f"home_ga_{w}"]  - base[f"away_ga_{w}"]
    base = base.dropna(subset=["home_ppg_5","away_ppg_5","home_g    base = base.dropna(subset=["home_ppg_5","away_ppg_5","home_gf_5","away_gf_5","home_ga_5","away_ga_5"])
