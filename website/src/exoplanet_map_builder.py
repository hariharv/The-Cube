
import argparse
import os
import math
import pandas as pd
import numpy as np
import plotly.express as px

# --------- Column normalization helpers ---------
def read_exoplanet_csv(path: str, mission_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    # normalize columns (case-insensitive lookup)
    cols = {c.lower(): c for c in df.columns}

    def pick(*candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    ra_col  = pick("ra", "ra_deg", "raj2000", "ra_str", "ra_deg_str")
    dec_col = pick("dec", "dec_deg", "decj2000", "dec_str", "decl", "dec_deg_str")
    if ra_col is None or dec_col is None:
        raise ValueError(f"Could not find RA/Dec in {os.path.basename(path)}. Columns: {list(df.columns)}")

    name_col = pick("pl_name", "kepoi_name", "koi_name", "planet_name", "hostname", "tic_id", "toi", "full_name", "loc_rowid", "kepid")

    # planet/star props
    rad_col  = pick("koi_prad", "pl_rade", "planet_radius", "planet_radius_earth", "rad", "radius_re", "pl_radj")
    per_col  = pick("koi_period", "pl_orbper", "period", "orbital_period", "orbper")
    teq_col  = pick("koi_teq", "pl_eqt", "teq", "eq_temp")
    teff_col = pick("koi_steff", "st_teff", "teff", "stellar_teff")
    snr_col  = pick("koi_model_snr", "snr")
    mag_col  = pick("koi_kepmag", "st_vmag", "phot_g_mean_mag", "vmag", "kepmag")

    # disposition variants
    dispo_col = pick("koi_pdisposition", "koi_disposition", "disposition", "toi_disposition", "tfopwg_disposition", "k2_disposition", "planetdisposition")

    out = pd.DataFrame()
    out["ra_deg"]   = pd.to_numeric(df[ra_col], errors="coerce")
    out["dec_deg"]  = pd.to_numeric(df[dec_col], errors="coerce")
    out["name"]     = df[name_col] if name_col else np.nan
    out["radius_re"] = pd.to_numeric(df[rad_col], errors="coerce") if rad_col else np.nan
    out["period_d"]  = pd.to_numeric(df[per_col], errors="coerce") if per_col else np.nan
    out["teq_k"]     = pd.to_numeric(df[teq_col], errors="coerce") if teq_col else np.nan
    out["teff_k"]    = pd.to_numeric(df[teff_col], errors="coerce") if teff_col else np.nan
    out["snr"]       = pd.to_numeric(df[snr_col], errors="coerce") if snr_col else np.nan
    out["mag"]       = pd.to_numeric(df[mag_col], errors="coerce") if mag_col else np.nan

    if dispo_col and dispo_col in df.columns:
        disposition_raw = df[dispo_col].astype(str)
    else:
        disposition_raw = pd.Series([""] * len(df))

    disp_norm = disposition_raw.str.upper().str.strip().replace(
        {"CP": "CONFIRMED", "PC": "CANDIDATE", "FP": "FALSE POSITIVE"}
    )
    out["disposition"] = disp_norm
    out["mission"] = mission_name

    # ensure numeric
    out = out.dropna(subset=["ra_deg", "dec_deg"])
    return out

def is_confirmed(text) -> bool:
    if not isinstance(text, str):
        return False
    return "CONFIRMED" in text.upper()

# --------- Weighted score & model results joining ---------
def compute_weighted_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    # Normalize each feature before weighting (robust to outliers)
    # Features: radius_re, period_d, teq_k, snr, mag (if present)
    features = {
        "radius": ("radius_re", 1.0),
        "period": ("period_d", 1.0),
        "teq":    ("teq_k", 1.0),
        "snr":    ("snr", 1.0),
        "mag":    ("mag", 1.0),
    }
    # Z-score normalize with clipping
    total = pd.Series(0.0, index=df.index, dtype=float)
    any_weight = False
    for key, (col, _) in features.items():
        w = float(weights.get(key, 0.0))
        if w == 0 or col not in df.columns:
            continue
        x = df[col].astype(float)
        mu = np.nanmean(x)
        sig = np.nanstd(x)
        if not np.isfinite(sig) or sig == 0:
            z = pd.Series(0.0, index=x.index, dtype=float)
        else:
            z = (x - mu) / sig
        z = z.clip(-3, 3)  # tame extremes
        total = total + w * z
        any_weight = True
    if not any_weight:
        return pd.Series(np.nan, index=df.index, dtype=float)
    # Min-max to [0,1] for nice color mapping
    v = total.replace([np.inf, -np.inf], np.nan)
    v_min, v_max = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_min == v_max:
        return pd.Series(0.5, index=df.index, dtype=float)
    return (v - v_min) / (v_max - v_min)

def load_model_scores(path: str, score_col: str = "score") -> pd.DataFrame:
    m = pd.read_csv(path, comment="#")
    # normalize column names
    cols = {c.lower(): c for c in m.columns}
    def pick(*candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None
    name_col = pick("name", "pl_name", "planet_name", "target", "id")
    ra_col   = pick("ra", "ra_deg")
    dec_col  = pick("dec", "dec_deg")
    if score_col not in m.columns:
        raise ValueError(f"Model results missing score column '{score_col}'. Columns: {list(m.columns)}")
    out = pd.DataFrame()
    if name_col:
        out["name"] = m[name_col].astype(str)
    if ra_col and dec_col:
        out["ra_deg"]  = pd.to_numeric(m[ra_col], errors="coerce")
        out["dec_deg"] = pd.to_numeric(m[dec_col], errors="coerce")
    out[score_col] = pd.to_numeric(m[score_col], errors="coerce")
    return out

def join_model_scores(df_all: pd.DataFrame, model_df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    df = df_all.copy()
    # 1) Try exact name match
    if "name" in model_df.columns and model_df["name"].notna().any():
        df["name_key"] = df["name"].astype(str).str.strip().str.upper()
        model_df["name_key"] = model_df["name"].astype(str).str.strip().str.upper()
        df = df.merge(model_df[["name_key", score_col]].dropna(subset=[score_col]), on="name_key", how="left")
        df = df.drop(columns=["name_key"])
    # 2) Fallback: approximate RA/Dec match by rounding
    if score_col not in df.columns or df[score_col].isna().all():
        if "ra_deg" in model_df.columns and "dec_deg" in model_df.columns:
            tmp_df = model_df.copy()
            tmp_df["ra_r"]  = tmp_df["ra_deg"].round(4)
            tmp_df["dec_r"] = tmp_df["dec_deg"].round(4)
            df["ra_r"]  = df["ra_deg"].round(4)
            df["dec_r"] = df["dec_deg"].round(4)
            df = df.merge(tmp_df[["ra_r", "dec_r", score_col]].dropna(subset=[score_col]), on=["ra_r", "dec_r"], how="left")
            df = df.drop(columns=["ra_r", "dec_r"])
    return df

# --------- Plotting ---------
def build_plot(df_all: pd.DataFrame,
               size_source: str = "radius_re",
               color_mode: str = "confirmed",
               score_series: pd.Series | None = None,
               output_html: str = "exoplanet_map.html") -> None:

    df = df_all.copy()
    # marker sizes
    if size_source == "weighted" and score_series is not None:
        size = score_series.fillna(0.5) * 12.0  # scale 0..1 -> up to 12
    elif size_source in df.columns:
        size = df[size_source].astype(float).fillna(3.0).clip(1, 12)
    else:
        size = pd.Series(6.0, index=df.index)

    hover_data = {
        "Mission": df["mission"],
        "Target Name": df["name"],
        "Disposition": df["disposition"],
        "Radius (R⊕)": df["radius_re"],
        "Period (days)": df["period_d"],
        "Teq (K)": df["teq_k"],
        "Star Teff (K)": df["teff_k"],
        "SNR": df["snr"],
        "Mag": df["mag"],
    }

    if color_mode == "confirmed":
        df["confirmed"] = df["disposition"].apply(is_confirmed)
        color = "confirmed"
        color_kwargs = dict(color_discrete_map={True: "red", False: "lightgray"})
    elif color_mode == "score" and score_series is not None:
        df["score"] = score_series
        color = "score"
        color_kwargs = dict(color_continuous_scale="Viridis")
    else:
        color = "mission"
        color_kwargs = dict()

    fig = px.scatter(
        df,
        x="ra_deg",
        y="dec_deg",
        size=size,
        color=color,
        hover_name="name",
        hover_data=hover_data,
        labels={"ra_deg": "Right Ascension (deg)", "dec_deg": "Declination (deg)"},
        **color_kwargs
    )
    fig.update_layout(
        title="Exoplanet Sky Map (RA–Dec)",
        xaxis_title="RA (deg)",
        yaxis_title="Dec (deg)",
        xaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=750,
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Wrote {output_html}")

# --------- CLI ---------
def parse_weights(weight_kv: list[str]) -> dict:
    weights = {}
    for item in weight_kv:
        if "=" not in item:
            raise ValueError(f"Bad --weights item: {item}. Use like radius=0.5")
        k, v = item.split("=", 1)
        weights[k.strip().lower()] = float(v.strip())
    return weights

def main():
    ap = argparse.ArgumentParser(description="Build interactive exoplanet sky map with custom weights and/or model scores.")
    ap.add_argument("--kepler", required=True, help="Path to Kepler CSV (e.g., data.csv)")
    ap.add_argument("--k2", required=False, help="Path to K2 CSV")
    ap.add_argument("--tess", required=False, help="Path to TESS CSV")
    ap.add_argument("--out", default="exoplanet_map.html", help="Output HTML path")

    # Weighting options (if no model results provided)
    ap.add_argument("--weights", nargs="*", default=[], help="Key=val pairs: radius=..., period=..., teq=..., snr=..., mag=...")
    ap.add_argument("--score-dest", choices=["size", "color", "both", "none"], default="none",
                    help="Where to apply the weighted score (if provided): size/color/both/none")

    # Model results join
    ap.add_argument("--model-results", help="CSV with model scores to join (by name or RA/Dec)")
    ap.add_argument("--score-col", default="score", help="Column name in model-results with the score")

    args = ap.parse_args()

    datasets = []
    datasets.append(read_exoplanet_csv(args.kepler, "Kepler"))
    if args.k2 and os.path.exists(args.k2):
        datasets.append(read_exoplanet_csv(args.k2, "K2"))
    if args.tess and os.path.exists(args.tess):
        datasets.append(read_exoplanet_csv(args.tess, "TESS"))
    df_all = pd.concat(datasets, ignore_index=True).dropna(subset=["ra_deg", "dec_deg"])

    # Determine score input: model file or weighted features
    score_series = None
    if args.model_results and os.path.exists(args.model_results):
        model_df = load_model_scores(args.model_results, score_col=args.score_col)
        df_all = join_model_scores(df_all, model_df, score_col=args.score_col)
        if args.score_col in df_all.columns and not df_all[args.score_col].isna().all():
            score_series = df_all[args.score_col].astype(float)
    if score_series is None and args.weights:
        weights = parse_weights(args.weights)
        score_series = compute_weighted_score(df_all, weights)

    # Decide plotting modes
    if args.score_dest in ("size", "both") and score_series is not None:
        size_source = "weighted"
    else:
        size_source = "radius_re"

    if args.score_dest in ("color", "both") and score_series is not None:
        color_mode = "score"
    else:
        color_mode = "confirmed"  # default: confirmed red, others gray

    build_plot(df_all, size_source=size_source, color_mode=color_mode, score_series=score_series, output_html=args.out)

if __name__ == "__main__":
    main()
