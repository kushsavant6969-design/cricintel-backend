import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
import pulp as pl
import os, tempfile, subprocess

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CricIntel",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1b2a 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #e0e6ef !important; }

.cricintel-banner {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 60%, #061a30 100%);
    border: 1px solid #00d4ff33;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.cricintel-banner h1 {
    color: #00d4ff;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    margin: 0;
    text-shadow: 0 0 20px #00d4ff66;
}
.cricintel-banner p { color: #7ba7c4; margin: 0.3rem 0 0; font-size: 0.95rem; }

.section-header {
    background: linear-gradient(90deg, #00d4ff15, transparent);
    border-left: 3px solid #00d4ff;
    padding: 0.5rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1.5rem 0 0.8rem;
    font-size: 1.05rem;
    font-weight: 600;
    color: #c8e6f5;
    letter-spacing: 0.04em;
}

.mapper-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.mapper-card .mc-title { font-size: 0.85rem; font-weight: 600; color: #00d4ff; margin-bottom: 0.4rem; }
.mapper-card .mc-sub { font-size: 0.75rem; color: #7ba7c4; }

.confidence-high { color: #4ade80; font-weight: 600; }
.confidence-med  { color: #fbbf24; font-weight: 600; }
.confidence-low  { color: #f87171; font-weight: 600; }

.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; letter-spacing:0.06em; }
.badge-bat  { background:#1a3a2a; color:#4ade80; border:1px solid #4ade8044; }
.badge-bowl { background:#1a1a3a; color:#818cf8; border:1px solid #818cf844; }
.badge-ar   { background:#2a2a1a; color:#fbbf24; border:1px solid #fbbf2444; }
.badge-wk   { background:#2a1a1a; color:#f87171; border:1px solid #f8717144; }

.player-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.player-card .pname { font-size: 1rem; font-weight: 600; color: #e0e6ef; }
.player-card .pstat { font-size: 0.8rem; color: #7ba7c4; margin-top: 0.2rem; }

.explain-card {
    background: #080f1a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.explain-card .ename { font-size: 0.95rem; font-weight: 600; color: #00d4ff; }
.explain-card .escores { font-size: 0.78rem; color: #7ba7c4; margin-top: 0.15rem; }
.explain-pos { color: #4ade80; font-size: 0.82rem; }
.explain-neg { color: #f87171; font-size: 0.82rem; }

.filter-panel {
    background: #080f1a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

.cricdiv { border:none; border-top:1px solid #1e3a5f; margin:1.2rem 0; }

.stDownloadButton button {
    background: linear-gradient(135deg, #00d4ff22, #0066aa22) !important;
    border: 1px solid #00d4ff55 !important;
    color: #00d4ff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-list"] { background:#0a0f1e; border-radius:8px; padding:4px; }
.stTabs [data-baseweb="tab"] { color:#7ba7c4 !important; font-weight:500; }
.stTabs [aria-selected="true"] { background:#0d2137 !important; color:#00d4ff !important; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# INTELLIGENT COLUMN MAPPER
# ═══════════════════════════════════════════════════════

# Known aliases for each internal field
COLUMN_ALIASES = {
    # Identity
    "player_id":   ["player_id","id","player id","uid","player_uid","playerid","p_id","pid","player_no","number"],
    "player":      ["player","name","player_name","full_name","fullname","playername","player name","cricket_name"],

    # Role
    "role":        ["role","position","player_role","type","player_type","category","playing_role","role_type"],

    # Age
    "age":         ["age","player_age","years","age_years","current_age"],

    # Batting
    "bat_hand":    ["bat_hand","batting_hand","hand","batting_style","dominant_hand","bat_style","batting_arm"],
    "runs":        ["runs","total_runs","runs_scored","batting_runs","career_runs","run_total"],
    "strike_rate": ["strike_rate","sr","batting_sr","bat_sr","strike rate","batting_strike_rate","sr_bat"],
    "boundary_pct":["boundary_pct","boundary_percent","boundary_%","boundaries_pct","boundary_rate"],
    "dot_ball_pct":["dot_ball_pct","dot_pct","dot_%","dot_ball_percent","dot_rate","dots_pct"],

    # Bowling
    "wickets":     ["wickets","wkts","total_wickets","bowling_wickets","career_wickets","wkt_total"],
    "economy":     ["economy","eco","economy_rate","bowling_economy","econ","bowl_eco","er"],
    "bowl_type":   ["bowl_type","bowling_type","bowling_style","bowl_style","bowling_arm_type","bowler_type"],

    # Matches
    "matches":     ["matches","games","total_matches","appearances","caps","career_matches","match_count"],

    # Flags
    "is_spinner":  ["is_spinner","spinner","spin_bowler","is_spin","spins"],
    "is_pacer":    ["is_pacer","pacer","pace_bowler","is_pace","paces","fast_bowler"],
    "is_overseas": ["is_overseas","overseas","foreign","international","is_foreign","non_domestic"],

    # Phase stats
    "pp_sr":       ["pp_sr","powerplay_sr","pp_strike_rate","powerplay_strike_rate","pp_batting_sr"],
    "middle_sr":   ["middle_sr","middle_overs_sr","mid_sr","middle_strike_rate"],
    "death_sr":    ["death_sr","death_overs_sr","death_strike_rate","finishing_sr"],
    "pp_runs":     ["pp_runs","powerplay_runs","pp_batting_runs"],
    "death_runs":  ["death_runs","death_overs_runs","finishing_runs"],
    "pp_eco":      ["pp_eco","powerplay_economy","pp_economy","powerplay_eco"],
    "middle_eco":  ["middle_eco","middle_economy","mid_eco","middle_overs_economy"],
    "death_eco":   ["death_eco","death_economy","death_overs_economy","finishing_eco"],
    "pp_wkts":     ["pp_wkts","powerplay_wickets","pp_wickets"],
    "death_wkts":  ["death_wkts","death_wickets","death_overs_wickets"],

    # Risk
    "injury_risk": ["injury_risk","injury","risk","injury_score","fit_risk","physical_risk"],

    # Format fit
    "county_red_ball_fit":   ["county_red_ball_fit","red_ball_fit","red_ball","redball_fit","first_class_fit"],
    "county_white_ball_fit": ["county_white_ball_fit","white_ball_fit","white_ball","whiteball_fit","limited_overs_fit"],
    "format_specialism":     ["format_specialism","format","specialism","format_type","best_format"],

    # Scouting
    "scouting_grade":        ["scouting_grade","grade","scout_grade","talent_grade","rating_grade"],
    "analyst_recommendation":["analyst_recommendation","recommendation","scout_recommendation","verdict","analyst_verdict"],

    # Auction/contract
    "current_salary_lakh":   ["current_salary_lakh","salary","wage","wages","annual_salary","base_price",
                               "salary_lakh","contract_value","fee","annual_fee","price","current_price"],
    "budget_lakh":           ["budget_lakh","budget","total_budget","squad_budget","purse","available_budget"],
    "max_players":           ["max_players","squad_size","max_squad","squad_max","team_size"],
    "min_bat":               ["min_bat","min_batters","min_batting","batters_min"],
    "min_bowl":              ["min_bowl","min_bowlers","min_bowling","bowlers_min"],
    "min_ar":                ["min_ar","min_allrounders","min_all_rounders","allrounders_min"],
    "min_wk":                ["min_wk","min_keepers","min_wicketkeepers","keepers_min"],
}

# Role value standardisation
ROLE_MAP = {
    "bat": "BAT", "batter": "BAT", "batsman": "BAT", "batting": "BAT", "b": "BAT",
    "bat/wk": "WK", "wk": "WK", "keeper": "WK", "wicketkeeper": "WK", "wk-bat": "WK",
    "wicket keeper": "WK", "keeper-batsman": "WK", "wkt": "WK",
    "bowl": "BOWL", "bowler": "BOWL", "bowling": "BOWL", "bwl": "BOWL",
    "pace bowler": "BOWL", "spin bowler": "BOWL", "fast bowler": "BOWL",
    "ar": "AR", "all-rounder": "AR", "allrounder": "AR", "all rounder": "AR",
    "all_rounder": "AR", "batting allrounder": "AR", "bowling allrounder": "AR",
}

BAT_HAND_MAP = {
    "l": "L", "lhb": "L", "left": "L", "left-hand": "L", "left hand": "L", "lh": "L",
    "r": "R", "rhb": "R", "right": "R", "right-hand": "R", "right hand": "R", "rh": "R",
}


def fuzzy_match_column(col_name: str, aliases: list) -> float:
    """Returns confidence score 0-1 for how well col_name matches aliases."""
    col_clean = col_name.lower().strip().replace(" ", "_").replace("-", "_")
    for alias in aliases:
        alias_clean = alias.lower().strip().replace(" ", "_").replace("-", "_")
        if col_clean == alias_clean:
            return 1.0
        if col_clean in alias_clean or alias_clean in col_clean:
            return 0.8
    return 0.0


def auto_detect_columns(df: pd.DataFrame) -> dict:
    """
    Automatically detects which columns in df map to internal field names.
    Returns: {internal_field: (detected_col, confidence)}
    """
    results = {}
    df_cols = list(df.columns)

    for field, aliases in COLUMN_ALIASES.items():
        best_col = None
        best_conf = 0.0
        for col in df_cols:
            conf = fuzzy_match_column(col, aliases)
            if conf > best_conf:
                best_conf = conf
                best_col = col
        if best_conf > 0:
            results[field] = (best_col, best_conf)

    return results


def standardise_roles(series: pd.Series) -> pd.Series:
    """Standardise role values to BAT/BOWL/AR/WK."""
    def map_role(val):
        if pd.isna(val):
            return "BAT"
        v = str(val).lower().strip()
        return ROLE_MAP.get(v, "BAT")
    return series.apply(map_role)


def standardise_bat_hand(series: pd.Series) -> pd.Series:
    """Standardise batting hand to L/R."""
    def map_hand(val):
        if pd.isna(val):
            return "R"
        v = str(val).lower().strip()
        return BAT_HAND_MAP.get(v, "R")
    return series.apply(map_hand)


def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Renames detected columns to internal names.
    mapping = {internal_field: source_col}
    """
    out = df.copy()
    rename_map = {}
    for internal, source in mapping.items():
        if source and source in out.columns and source != internal:
            rename_map[source] = internal
    out = out.rename(columns=rename_map)

    # Standardise role values
    if "role" in out.columns:
        out["role"] = standardise_roles(out["role"])

    # Standardise bat_hand
    if "bat_hand" in out.columns:
        out["bat_hand"] = standardise_bat_hand(out["bat_hand"])

    return out


def render_mapping_summary(detected: dict, required_fields: list):
    """Show a clean summary of what was detected."""
    found = [f for f in required_fields if f in detected]
    missing = [f for f in required_fields if f not in detected]
    optional_found = [f for f in detected if f not in required_fields]

    conf_levels = {"high": [], "medium": [], "low": []}
    for f, (col, conf) in detected.items():
        if conf >= 0.9:
            conf_levels["high"].append(f)
        elif conf >= 0.7:
            conf_levels["medium"].append(f)
        else:
            conf_levels["low"].append(f)

    total = len(required_fields)
    found_n = len(found)
    pct = int((found_n / total) * 100) if total > 0 else 0

    if pct == 100:
        conf_class = "confidence-high"
        conf_label = "✅ All required fields detected"
    elif pct >= 70:
        conf_class = "confidence-med"
        conf_label = f"⚠️ {found_n}/{total} required fields detected"
    else:
        conf_class = "confidence-low"
        conf_label = f"❌ Only {found_n}/{total} required fields detected"

    st.markdown(f"""
    <div class="mapper-card">
        <div class="mc-title">🧠 Intelligent Column Detection</div>
        <div class="{conf_class}" style="font-size:0.9rem; margin-bottom:0.5rem;">{conf_label}</div>
        <div class="mc-sub">
            Required detected: <b>{found_n}/{total}</b> &nbsp;|&nbsp;
            Optional detected: <b>{len(optional_found)}</b> &nbsp;|&nbsp;
            Missing (will be inferred): <b>{len(missing)}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if missing:
        with st.expander("ℹ️ Missing fields — will be inferred automatically"):
            for m in missing:
                st.markdown(f"• `{m}` — will be estimated from available data")

    if conf_levels["medium"] or conf_levels["low"]:
        with st.expander("⚠️ Low-confidence detections — verify these"):
            for f in conf_levels["medium"] + conf_levels["low"]:
                col, conf = detected[f]
                st.markdown(f"• `{f}` ← mapped from `{col}` (confidence: {conf:.0%})")


def smart_merge(players_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently merges two dataframes by finding the common ID column.
    """
    # Detect ID columns in both
    players_detected = auto_detect_columns(players_df)
    perf_detected    = auto_detect_columns(perf_df)

    p_id_col = players_detected.get("player_id", (None, 0))[0]
    r_id_col = perf_detected.get("player_id", (None, 0))[0]

    # If same column name found in both — use it directly
    if p_id_col and r_id_col:
        # Rename both to player_id for consistent merge
        if p_id_col != "player_id":
            players_df = players_df.rename(columns={p_id_col: "player_id"})
        if r_id_col != "player_id":
            perf_df = perf_df.rename(columns={r_id_col: "player_id"})
        # FIX: cast both to string to avoid object vs int64 merge error
        players_df["player_id"] = players_df["player_id"].astype(str)
        perf_df["player_id"]    = perf_df["player_id"].astype(str)
        df = players_df.merge(perf_df, on="player_id", how="left", suffixes=("", "_perf"))
    else:
        # Try to find any common column
        common = set(players_df.columns) & set(perf_df.columns)
        if common:
            join_col = list(common)[0]
            df = players_df.merge(perf_df, on=join_col, how="left", suffixes=("", "_perf"))
            df = df.rename(columns={join_col: "player_id"})
        else:
            # Last resort — join by index
            st.warning("⚠️ No common ID column found. Joining by row order.")
            df = pd.concat([players_df.reset_index(drop=True),
                            perf_df.reset_index(drop=True)], axis=1)
            df["player_id"] = df.index + 1

    # Resolve age conflict
    if "age" not in df.columns:
        if "age_perf" in df.columns:
            df["age"] = df["age_perf"]
        elif "age" in perf_df.columns:
            df["age"] = perf_df["age"].values[:len(df)]
        else:
            df["age"] = 25

    return df


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════
def norm01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    mn, mx = s.min(), s.max()
    if mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)

def clamp01(x):
    return np.clip(x, 0, 1)

def to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8")

def stable_noise(id_series: pd.Series, mod=97) -> np.ndarray:
    s = pd.to_numeric(id_series, errors="coerce").fillna(0).astype(int)
    return (s % mod) / float(mod)

def safe_numeric(df, cols, default=0.0):
    for c in cols:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    return df

def role_badge(role):
    cls = {"BAT":"badge-bat","BOWL":"badge-bowl","AR":"badge-ar","WK":"badge-wk"}.get(role,"badge-bat")
    return f'<span class="badge {cls}">{role}</span>'

def section(title, icon="▸"):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def cric_divider():
    st.markdown('<hr class="cricdiv">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# CORE DATA PIPELINE
# ═══════════════════════════════════════════════════════
def build_base_df(players: pd.DataFrame, perf: pd.DataFrame, contracts: pd.DataFrame | None = None):
    # ── Smart merge ────────────────────────────────────────────────────────
    df = smart_merge(players, perf)

    if contracts is not None:
        c_detected = auto_detect_columns(contracts)
        c_id = c_detected.get("player_id", (None, 0))[0]
        if c_id and c_id != "player_id":
            contracts = contracts.rename(columns={c_id: "player_id"})
        # FIX: cast both to string to avoid type mismatch
        df["player_id"] = df["player_id"].astype(str)
        contracts["player_id"] = contracts["player_id"].astype(str)
        df = df.merge(contracts, on="player_id", how="left", suffixes=("", "_contract"))

    # ── Apply intelligent column mapping ──────────────────────────────────
    detected = auto_detect_columns(df)
    mapping  = {field: col for field, (col, _) in detected.items()}
    df = apply_column_mapping(df, mapping)

    # ── Required column check ─────────────────────────────────────────────
    if "player" not in df.columns:
        name_candidates = [c for c in df.columns if "name" in c.lower()]
        if name_candidates:
            df["player"] = df[name_candidates[0]]
        else:
            df["player"] = ["Player_" + str(i) for i in df.index]

    if "player_id" not in df.columns:
        df["player_id"] = df.index + 1

    # ── Numeric safety ────────────────────────────────────────────────────
    df = safe_numeric(df, ["matches","runs","strike_rate","wickets","economy",
                           "dot_ball_pct","boundary_pct","age"], 0.0)

    # ── Optional phase columns ────────────────────────────────────────────
    for col in ["pp_sr","middle_sr","death_sr","pp_runs","death_runs",
                "pp_eco","middle_eco","death_eco","pp_wkts","death_wkts",
                "injury_risk","availability_risk"]:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["is_pp_bowler","is_death_bowler","is_pp_batter","is_death_hitter",
                "is_spinner","is_pacer","is_overseas"]:
        if col not in df.columns:
            df[col] = 0

    if "bat_hand"    not in df.columns: df["bat_hand"]    = "R"
    if "bowling_arm" not in df.columns: df["bowling_arm"] = np.nan
    if "spin_type"   not in df.columns: df["spin_type"]   = np.nan

    # ── Phase proxies ─────────────────────────────────────────────────────
    noise = stable_noise(df["player_id"])

    if df["pp_sr"].isna().all():
        df["pp_sr"]      = df["strike_rate"] + (-2  + 6*(noise-0.5))
        df["middle_sr"]  = df["strike_rate"] + (-6  + 8*(noise-0.5))
        df["death_sr"]   = df["strike_rate"] + (8   + 10*(noise-0.5))
        df["pp_runs"]    = (df["runs"] * (0.33+0.06*(noise-0.5))).astype(float)
        df["death_runs"] = (df["runs"] * (0.23+0.06*(noise-0.5))).astype(float)

    if df["pp_eco"].isna().all():
        df["pp_eco"]     = df["economy"] + (-0.25+0.6*(noise-0.5))
        df["middle_eco"] = df["economy"] + (0.00 +0.5*(noise-0.5))
        df["death_eco"]  = df["economy"] + (0.55 +0.8*(noise-0.5))
        df["pp_wkts"]    = (df["wickets"]*(0.33+0.05*(noise-0.5))).astype(float)
        df["death_wkts"] = (df["wickets"]*(0.23+0.05*(noise-0.5))).astype(float)

    # ── Role inference ────────────────────────────────────────────────────
    if "batting_role" not in df.columns: df["batting_role"] = "ANCHOR"
    if "bowling_role" not in df.columns: df["bowling_role"] = "MIDDLE"

    is_bat  = df["role"].isin(["BAT","AR","WK"])
    is_bowl = df["role"].isin(["BOWL","AR"])

    df.loc[is_bat, "batting_role"] = "ANCHOR"
    df.loc[is_bat & (df["pp_sr"]    >= df["pp_sr"].quantile(0.70)),    "batting_role"] = "OPENER"
    df.loc[is_bat & (df["death_sr"] >= df["death_sr"].quantile(0.75)), "batting_role"] = "FINISHER"

    df.loc[is_bowl, "bowling_role"] = "MIDDLE"
    df.loc[is_bowl & ((df["pp_wkts"]    >= df["pp_wkts"].quantile(0.65))    | (df["is_pp_bowler"].astype(int)==1)),    "bowling_role"] = "PP"
    df.loc[is_bowl & ((df["death_wkts"] >= df["death_wkts"].quantile(0.70)) | (df["is_death_bowler"].astype(int)==1)), "bowling_role"] = "DEATH"

    df["is_opener"]        = (df["batting_role"]=="OPENER").astype(int)
    df["is_finisher"]      = (df["batting_role"]=="FINISHER").astype(int)
    df["is_death_bowler2"] = (df["bowling_role"]=="DEATH").astype(int)
    df["is_pp_bowler2"]    = (df["bowling_role"]=="PP").astype(int)
    df["is_top4_candidate"]= (
        df["role"].isin(["BAT","WK","AR"]) &
        ((df["is_opener"]==1) | (df["runs"] >= df["runs"].quantile(0.60)))
    ).astype(int)

    # ── Bowling arm + spin type ───────────────────────────────────────────
    if df["bowling_arm"].isna().all():
        df["bowling_arm"] = np.where(noise < 0.28, "L", "R")
    df["bowling_arm"] = df["bowling_arm"].astype(str).str.upper()
    df.loc[~df["bowling_arm"].isin(["L","R"]), "bowling_arm"] = "R"

    if df["spin_type"].isna().all():
        df["spin_type"] = "NONE"
        spin_mask = df["is_spinner"].astype(int)==1
        df.loc[spin_mask, "spin_type"] = np.where(noise[spin_mask]<0.55, "OFF", "LEG")
    df["spin_type"] = df["spin_type"].astype(str).str.upper()

    df["is_left_arm_spinner"]  = ((df["is_spinner"].astype(int)==1) & (df["bowling_arm"]=="L")).astype(int)
    df["is_right_arm_spinner"] = ((df["is_spinner"].astype(int)==1) & (df["bowling_arm"]=="R")).astype(int)
    df["is_off_spinner"]       = ((df["is_spinner"].astype(int)==1) & (df["spin_type"]=="OFF")).astype(int)
    df["is_leg_spinner"]       = ((df["is_spinner"].astype(int)==1) & (df["spin_type"]=="LEG")).astype(int)
    df["is_left_arm_pacer"]    = ((df["is_pacer"].astype(int)==1)   & (df["bowling_arm"]=="L")).astype(int)
    df["is_right_arm_pacer"]   = ((df["is_pacer"].astype(int)==1)   & (df["bowling_arm"]=="R")).astype(int)

    # ── Risk model ────────────────────────────────────────────────────────
    if df["injury_risk"].isna().all():
        df["injury_risk"] = clamp01(0.12 + 0.02*(df["age"]-df["age"].min()) + 0.20*norm01(df["matches"]) + 0.10*df["is_pacer"].astype(float))
    if df["availability_risk"].isna().all():
        df["availability_risk"] = clamp01(0.08 + 0.15*df["is_overseas"].astype(float) + 0.10*norm01(df["matches"]))
    df["total_risk"] = clamp01(0.65*df["injury_risk"].astype(float) + 0.35*df["availability_risk"].astype(float))

    return df


def compute_phase_scores(df, w_pp=1.0, w_mid=1.0, w_death=1.2):
    pp_bat    = 0.6*norm01(df["pp_sr"])    + 0.4*norm01(df["pp_runs"])
    mid_bat   = norm01(df["middle_sr"])
    death_bat = 0.6*norm01(df["death_sr"]) + 0.4*norm01(df["death_runs"])
    pp_bowl   = 0.6*norm01(df["pp_wkts"])  + 0.4*(1-norm01(df["pp_eco"]))
    mid_bowl  = 0.7*(1-norm01(df["middle_eco"])) + 0.3*norm01(df["dot_ball_pct"])
    death_bowl= 0.6*norm01(df["death_wkts"])+ 0.4*(1-norm01(df["death_eco"]))

    df["pp_bat_score"]    = pp_bat
    df["mid_bat_score"]   = mid_bat
    df["death_bat_score"] = death_bat
    df["pp_bowl_score"]   = pp_bowl
    df["mid_bowl_score"]  = mid_bowl
    df["death_bowl_score"]= death_bowl

    impact = []
    for r, b, w in zip(
        df["role"].fillna("BAT"),
        (w_pp*pp_bat + w_mid*mid_bat + w_death*death_bat),
        (w_pp*pp_bowl + w_mid*mid_bowl + w_death*death_bowl)
    ):
        if r=="BAT":   impact.append(b)
        elif r=="BOWL":impact.append(w)
        elif r=="AR":  impact.append(0.55*b+0.45*w)
        elif r=="WK":  impact.append(0.95*b+0.05)
        else:          impact.append(b)

    df["match_impact_score"] = norm01(pd.Series(impact, index=df.index))
    return df


# ═══════════════════════════════════════════════════════
# SIMILARITY SEARCH
# ═══════════════════════════════════════════════════════
def get_similar_players(df_all, target_name, top_k=12):
    trow = df_all[df_all["player"]==target_name].head(1)
    if len(trow)==0:
        return pd.DataFrame()

    feat_cols = [c for c in [
        "pp_bat_score","mid_bat_score","death_bat_score",
        "pp_bowl_score","mid_bowl_score","death_bowl_score",
        "strike_rate","economy","dot_ball_pct","boundary_pct",
        "match_impact_score","total_risk"
    ] if c in df_all.columns]

    candidates = df_all[df_all["player"]!=target_name].copy()
    t_vec = trow[feat_cols].apply(pd.to_numeric,errors="coerce").fillna(0.0).values

    scaler = StandardScaler()
    cand_feats = candidates[feat_cols].apply(pd.to_numeric,errors="coerce").fillna(0.0).values
    scaled = scaler.fit_transform(np.vstack([cand_feats, t_vec]))
    cand_s = scaled[:-1]
    t_s    = scaled[-1].reshape(1,-1)

    sims = cosine_similarity(cand_s, t_s).reshape(-1)
    candidates = candidates.copy()
    candidates["similarity"] = sims

    target_role = trow["role"].values[0]
    return candidates[candidates["role"]==target_role].sort_values("similarity",ascending=False).head(top_k)


# ═══════════════════════════════════════════════════════
# OPTIMISERS
# ═══════════════════════════════════════════════════════
def optimize_squad_soft(df, budget_limit, max_players, min_role,
                        price_col, extra_min_flags=None, extra_max_flags=None,
                        lock_players=None, max_single_price=None,
                        penalty_weight=2.5, price_concentration_penalty=0.0):
    d = df.copy()
    players_list = d["player"].tolist()
    if not players_list:
        return pd.DataFrame(), {"status":"empty"}

    x = pl.LpVariable.dicts("pick", players_list, lowBound=0, upBound=1, cat="Binary")
    prob = pl.LpProblem("SquadSoft", pl.LpMaximize)

    slack = {}
    if extra_min_flags:
        for col, mn in extra_min_flags.items():
            if col in d.columns and int(mn)>0:
                slack[col] = pl.LpVariable(f"slack_{col}", lowBound=0, cat="Continuous")

    norm_price = norm01(d[price_col]).values
    prob += (
        pl.lpSum(d.loc[d.player==p,"objective_score"].values[0]*x[p] for p in players_list)
        - float(penalty_weight)*pl.lpSum(slack[c] for c in slack)
        - float(price_concentration_penalty)*pl.lpSum(norm_price[i]*x[p] for i,p in enumerate(players_list))
    )

    prob += pl.lpSum(d.loc[d.player==p,price_col].values[0]*x[p] for p in players_list) <= float(budget_limit)
    prob += pl.lpSum(x[p] for p in players_list) <= int(max_players)

    def role_count(code):
        return pl.lpSum(x[p] for p in players_list if d.loc[d.player==p,"role"].values[0]==code)
    for r, mn in min_role.items():
        prob += role_count(r) >= int(mn)

    if extra_min_flags:
        for col, mn in extra_min_flags.items():
            if col in d.columns and int(mn)>0:
                cnt = pl.lpSum(x[p] for p in players_list if int(d.loc[d.player==p,col].values[0])==1)
                prob += cnt+slack[col] >= int(mn)

    if extra_max_flags:
        for col, mx in extra_max_flags.items():
            if col in d.columns and mx is not None:
                prob += pl.lpSum(x[p] for p in players_list if int(d.loc[d.player==p,col].values[0])==1) <= int(mx)

    if max_single_price is not None:
        for p in players_list:
            prob += d.loc[d.player==p,price_col].values[0]*x[p] <= float(max_single_price)

    if lock_players:
        for lp in lock_players:
            if lp in players_list:
                prob += x[lp]==1

    prob.solve(pl.PULP_CBC_CMD(msg=False))
    picked = [p for p in players_list if x[p].value()==1]
    squad  = d[d.player.isin(picked)].copy()

    violations = {}
    for col in slack:
        try: violations[col] = float(slack[col].value())
        except: violations[col] = None

    metrics = {
        "status":"ok" if len(squad) else "no_solution",
        "count":int(len(squad)),
        "spend":float(squad[price_col].sum()) if len(squad) else 0.0,
        "objective":float(squad["objective_score"].sum()) if len(squad) else 0.0,
        "violations":violations
    }
    for r in ["BAT","BOWL","AR","WK"]:
        metrics[r] = int((squad["role"]==r).sum()) if len(squad) else 0
    return squad, metrics


def pick_best_xi(squad, xi_size, xi_min_role, max_overseas_xi, enforce_left_in_top4):
    if len(squad)==0:
        return pd.DataFrame(), {"status":"empty"}

    d = squad.copy()
    players_list = d["player"].tolist()
    x = pl.LpVariable.dicts("xi", players_list, lowBound=0, upBound=1, cat="Binary")
    prob = pl.LpProblem("BestXI", pl.LpMaximize)

    prob += pl.lpSum(d.loc[d.player==p,"xi_score"].values[0]*x[p] for p in players_list)
    prob += pl.lpSum(x[p] for p in players_list)==int(xi_size)

    def role_count(code):
        return pl.lpSum(x[p] for p in players_list if d.loc[d.player==p,"role"].values[0]==code)
    for r, mn in xi_min_role.items():
        prob += role_count(r) >= int(mn)

    if max_overseas_xi is not None and "is_overseas" in d.columns:
        prob += pl.lpSum(x[p] for p in players_list if int(d.loc[d.player==p,"is_overseas"].values[0])==1) <= int(max_overseas_xi)

    if enforce_left_in_top4 and "bat_hand" in d.columns and "is_top4_candidate" in d.columns:
        prob += pl.lpSum(
            x[p] for p in players_list
            if (int(d.loc[d.player==p,"is_top4_candidate"].values[0])==1 and
                str(d.loc[d.player==p,"bat_hand"].values[0])=="L")
        ) >= 1

    prob.solve(pl.PULP_CBC_CMD(msg=False))
    picked = [p for p in players_list if x[p].value()==1]
    xi = d[d.player.isin(picked)].copy()
    metrics = {"status":"ok" if len(xi) else "no_solution","count":int(len(xi)),
               "xi_score":float(xi["xi_score"].sum()) if len(xi) else 0.0}
    return xi, metrics


# ═══════════════════════════════════════════════════════
# SCOUT MODE
# ═══════════════════════════════════════════════════════
def run_scout_mode():
    # Read from unified session state
    df = st.session_state["df_master"].copy()

    # ── SNAPSHOT ──────────────────────────────────────────────────────────
    cric_divider()
    section("Squad Snapshot", "📊")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Players", len(df))
    c2.metric("Batters",       int((df["role"]=="BAT").sum()))
    c3.metric("Bowlers",       int((df["role"]=="BOWL").sum()))
    c4.metric("All-rounders",  int((df["role"]=="AR").sum()))
    c5.metric("Wicketkeepers", int((df["role"]=="WK").sum()))

    # ── CHARTS ────────────────────────────────────────────────────────────
    cric_divider()
    section("Role & Nationality Breakdown", "🌍")
    t1, t2 = st.columns(2)
    with t1:
        role_counts = df["role"].value_counts().reset_index()
        role_counts.columns = ["Role","Count"]
        st.bar_chart(role_counts.set_index("Role"), color="#00d4ff")
    with t2:
        if "country" in df.columns:
            cc = df["country"].value_counts().head(8).reset_index()
            cc.columns = ["Country","Count"]
            st.bar_chart(cc.set_index("Country"), color="#818cf8")

    # ── SMART FILTER PANEL ────────────────────────────────────────────────
    cric_divider()
    section("Smart Filter Panel", "🔎")

    with st.expander("🔽 Open Filters", expanded=True):
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)

        f1, f2, f3 = st.columns(3)

        # Role filter
        role_options = ["All"] + sorted(df["role"].dropna().unique().tolist())
        sel_role = f1.multiselect("Role", role_options[1:], default=[])

        # Age filter
        if df["age"].max() > 0:
            age_min = int(df["age"].min()) if df["age"].min() > 0 else 15
            age_max = int(df["age"].max()) if df["age"].max() > 0 else 45
            sel_age = f2.slider("Age range", age_min, age_max, (age_min, age_max))
        else:
            sel_age = (15, 45)

        # Batting hand
        hand_options = ["All","Left (L)","Right (R)"]
        sel_hand = f3.selectbox("Batting hand", hand_options)

        f4, f5, f6 = st.columns(3)

        # Bowling type
        bowl_options = ["All","Spinner","Pacer"]
        sel_bowl = f4.selectbox("Bowling type", bowl_options)

        # Batting role
        bat_role_opts = ["All"] + sorted(df["batting_role"].dropna().unique().tolist())
        sel_bat_role = f5.selectbox("Batting role", bat_role_opts)

        # Bowling role
        bowl_role_opts = ["All"] + sorted(df["bowling_role"].dropna().unique().tolist())
        sel_bowl_role = f6.selectbox("Bowling role", bowl_role_opts)

        f7, f8, f9 = st.columns(3)

        # Form trend filter
        if "form_trend" in df.columns:
            trend_opts = ["All"] + sorted(df["form_trend"].dropna().unique().tolist())
            sel_trend = f7.selectbox("Form trend", trend_opts)
        else:
            sel_trend = "All"

        # Scouting grade filter
        if "scouting_grade" in df.columns:
            grade_opts = ["All"] + sorted(df["scouting_grade"].dropna().unique().tolist())
            sel_grade = f8.selectbox("Scouting grade", grade_opts)
        else:
            sel_grade = "All"

        # Format specialism
        if "format_specialism" in df.columns:
            fmt_opts = ["All"] + sorted(df["format_specialism"].dropna().unique().tolist())
            sel_format = f8.selectbox("Format specialism", fmt_opts)
        else:
            sel_format = "All"

        # Analyst recommendation
        if "analyst_recommendation" in df.columns:
            rec_opts = ["All"] + sorted(df["analyst_recommendation"].dropna().unique().tolist())
            sel_rec = f9.selectbox("Analyst recommendation", rec_opts)
        else:
            sel_rec = "All"

        # Impact score slider
        fi1, fi2 = st.columns(2)
        sel_impact = fi1.slider("Min impact score", 0.0, 1.0, 0.0, 0.05)
        sel_risk   = fi2.slider("Max risk score",   0.0, 1.0, 1.0, 0.05)

        # Overseas filter
        sel_overseas = st.radio("Overseas status", ["All","Domestic only","Overseas only"], horizontal=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── APPLY FILTERS ─────────────────────────────────────────────────────
    filtered = df.copy()

    if sel_role:
        filtered = filtered[filtered["role"].isin(sel_role)]
    if sel_age != (int(df["age"].min()) if df["age"].min()>0 else 15,
                   int(df["age"].max()) if df["age"].max()>0 else 45):
        filtered = filtered[(filtered["age"]>=sel_age[0]) & (filtered["age"]<=sel_age[1])]
    if sel_hand == "Left (L)":
        filtered = filtered[filtered["bat_hand"]=="L"]
    elif sel_hand == "Right (R)":
        filtered = filtered[filtered["bat_hand"]=="R"]
    if sel_bowl == "Spinner":
        filtered = filtered[filtered["is_spinner"].astype(int)==1]
    elif sel_bowl == "Pacer":
        filtered = filtered[filtered["is_pacer"].astype(int)==1]
    if sel_bat_role != "All":
        filtered = filtered[filtered["batting_role"]==sel_bat_role]
    if sel_bowl_role != "All":
        filtered = filtered[filtered["bowling_role"]==sel_bowl_role]
    if sel_trend != "All" and "form_trend" in filtered.columns:
        filtered = filtered[filtered["form_trend"]==sel_trend]
    if sel_grade != "All" and "scouting_grade" in filtered.columns:
        filtered = filtered[filtered["scouting_grade"]==sel_grade]
    if sel_format != "All" and "format_specialism" in filtered.columns:
        filtered = filtered[filtered["format_specialism"]==sel_format]
    if sel_rec != "All" and "analyst_recommendation" in filtered.columns:
        filtered = filtered[filtered["analyst_recommendation"]==sel_rec]
    if sel_overseas == "Domestic only":
        filtered = filtered[filtered["is_overseas"].astype(int)==0]
    elif sel_overseas == "Overseas only":
        filtered = filtered[filtered["is_overseas"].astype(int)==1]

    filtered = filtered[filtered["match_impact_score"] >= sel_impact]
    filtered = filtered[filtered["total_risk"] <= sel_risk]

    st.caption(f"Showing **{len(filtered)}** of {len(df)} players")

    # ── TOP PROFILES TABLE ────────────────────────────────────────────────
    cric_divider()
    section("Top Profiles", "🏆")

    show_cols = [c for c in ["player","form_trend","role","age","country","bat_hand","bowling_arm",
                              "batting_role","bowling_role","matches","runs","strike_rate",
                              "wickets","economy","dot_ball_pct","boundary_pct",
                              "recent_matches","recent_sr","recent_economy","form_index",
                              "scouting_grade","format_specialism","analyst_recommendation",
                              "match_impact_score","total_risk"] if c in filtered.columns]

    st.dataframe(
        filtered[show_cols].sort_values("match_impact_score",ascending=False).head(50).style.format(
            {c:"{:.3f}" for c in ["match_impact_score","total_risk"] if c in show_cols} |
            {"strike_rate":"{:.1f}","economy":"{:.2f}"}
        ),
        use_container_width=True, height=440
    )

    # ── SHORTLIST FEATURE ─────────────────────────────────────────────────
    cric_divider()
    section("Shortlist", "📋")
    st.caption("Add players to your shortlist for export and comparison.")

    if "shortlist" not in st.session_state:
        st.session_state["shortlist"] = []

    all_player_names = filtered.sort_values("match_impact_score", ascending=False)["player"].tolist()

    sl1, sl2 = st.columns([3,1])
    add_players = sl1.multiselect(
        "Add players to shortlist",
        options=[p for p in all_player_names if p not in st.session_state["shortlist"]],
        default=[],
        key="shortlist_add"
    )

    if sl2.button("Add to Shortlist", type="primary"):
        for p in add_players:
            if p not in st.session_state["shortlist"]:
                st.session_state["shortlist"].append(p)
        st.rerun()

    if len(st.session_state["shortlist"]) == 0:
        st.info("No players shortlisted yet. Add players above.")
    else:
        st.markdown(f"**{len(st.session_state['shortlist'])} players shortlisted**")

        shortlist_df = df[df["player"].isin(st.session_state["shortlist"])].copy()

        sl_cols = [c for c in ["player","role","age","bat_hand","batting_role",
                                "bowling_role","matches","runs","strike_rate",
                                "wickets","economy","match_impact_score","total_risk"] if c in shortlist_df.columns]

        st.dataframe(
            shortlist_df[sl_cols].style.format(
                {"match_impact_score":"{:.3f}","total_risk":"{:.3f}",
                 "strike_rate":"{:.1f}","economy":"{:.2f}"}
            ),
            use_container_width=True,
            height=min(100 + len(shortlist_df)*35, 400)
        )

        remove_col, clear_col, export_col = st.columns(3)
        remove_player = remove_col.selectbox(
            "Remove a player",
            ["Select"] + st.session_state["shortlist"],
            key="shortlist_remove"
        )
        if remove_col.button("Remove"):
            if remove_player != "Select":
                st.session_state["shortlist"].remove(remove_player)
                st.rerun()

        if clear_col.button("Clear All"):
            st.session_state["shortlist"] = []
            st.rerun()

        export_col.download_button(
            "Export Shortlist CSV",
            to_csv_bytes(shortlist_df),
            "cricintel_shortlist.csv",
            "text/csv"
        )

    # ── PLAYER PROFILE CARD ───────────────────────────────────────────────
    cric_divider()
    section("Player Profile Card", "👤")
    st.caption("Search any player to see their full breakdown and radar comparison vs similar players.")

    search_col, _ = st.columns([2,1])
    profile_player = search_col.selectbox(
        "Search player name", ["— Select a player —"] + sorted(df["player"].tolist()),
        index=0, key="profile_select"
    )

    if profile_player != "— Select a player —":
        import math
        import streamlit.components.v1 as components

        prow = df[df["player"]==profile_player].iloc[0]
        sim_for_radar = get_similar_players(df, profile_player, top_k=3)

        # Header metrics
        h1,h2,h3,h4,h5 = st.columns(5)
        h1.metric("Role",     prow.get("role","—"))
        h2.metric("Age",      int(prow.get("age",0)) if float(prow.get("age",0))>0 else "—")
        h3.metric("Country",  prow.get("country","—") if "country" in df.columns else "—")
        h4.metric("Bat Hand", prow.get("bat_hand","—"))
        h5.metric("Matches",  int(prow.get("matches",0)))

        st.markdown(f"""
        <div class="player-card">
            <div class="pname">🏏 {profile_player} {role_badge(prow["role"])}</div>
            <div class="pstat">
                Runs: <b>{int(prow.get("runs",0))}</b> &nbsp;|&nbsp;
                SR: <b>{float(prow.get("strike_rate",0)):.1f}</b> &nbsp;|&nbsp;
                Wickets: <b>{int(prow.get("wickets",0))}</b> &nbsp;|&nbsp;
                Economy: <b>{float(prow.get("economy",0)):.2f}</b> &nbsp;|&nbsp;
                Dot Ball%: <b>{float(prow.get("dot_ball_pct",0)):.1f}</b> &nbsp;|&nbsp;
                Boundary%: <b>{float(prow.get("boundary_pct",0)):.1f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Scouting info
        si1,si2,si3 = st.columns(3)
        if "scouting_grade" in df.columns:
            si1.metric("Scouting Grade", prow.get("scouting_grade","—"))
        if "analyst_recommendation" in df.columns:
            si2.metric("Recommendation", prow.get("analyst_recommendation","—"))
        if "format_specialism" in df.columns:
            si3.metric("Format Specialism", str(prow.get("format_specialism","—")))

        # Format fit
        if "county_red_ball_fit" in df.columns or "county_white_ball_fit" in df.columns:
            ff1,ff2 = st.columns(2)
            if "county_red_ball_fit" in df.columns:
                ff1.metric("Red Ball Fit",   f"{float(prow.get('county_red_ball_fit',0)):.1f}/100")
            if "county_white_ball_fit" in df.columns:
                ff2.metric("White Ball Fit", f"{float(prow.get('county_white_ball_fit',0)):.1f}/100")

        # Radar chart
        st.markdown(f"**📡 {profile_player} — Radar vs Top 3 Similar Players**")

        radar_axes = [c for c in [
            "pp_bat_score","mid_bat_score","death_bat_score",
            "pp_bowl_score","mid_bowl_score","death_bowl_score",
            "match_impact_score"
        ] if c in df.columns]

        axis_labels = {
            "pp_bat_score":"PP Bat","mid_bat_score":"Mid Bat","death_bat_score":"Death Bat",
            "pp_bowl_score":"PP Bowl","mid_bowl_score":"Mid Bowl","death_bowl_score":"Death Bowl",
            "match_impact_score":"Impact"
        }
        labels = [axis_labels.get(c,c) for c in radar_axes]
        N = len(labels)
        cx, cy, r = 300, 270, 190
        colors = ["#00d4ff","#4ade80","#fbbf24","#f87171"]

        def get_vals(row):
            return [float(row.get(c,0)) for c in radar_axes]

        players_data = [(profile_player, get_vals(prow), colors[0])]
        for i, (_, srow) in enumerate(sim_for_radar.head(3).iterrows()):
            players_data.append((srow["player"], get_vals(srow), colors[i+1]))

        svg = ['<svg viewBox="0 0 600 570" xmlns="http://www.w3.org/2000/svg" style="background:#080f1a;border-radius:12px;border:1px solid #1e3a5f;width:100%;max-width:600px;">']

        # Grid rings
        for ring in [0.25,0.5,0.75,1.0]:
            pts = []
            for i in range(N):
                angle = (i/N)*2*math.pi - math.pi/2
                pts.append(f"{cx+ring*r*math.cos(angle):.1f},{cy+ring*r*math.sin(angle):.1f}")
            svg.append(f'<polygon points="{" ".join(pts)}" fill="none" stroke="#1e3a5f" stroke-width="1"/>')

        # Axes + labels
        for i, label in enumerate(labels):
            angle = (i/N)*2*math.pi - math.pi/2
            x2 = cx+r*math.cos(angle)
            y2 = cy+r*math.sin(angle)
            svg.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#1e3a5f" stroke-width="1"/>')
            lx = cx+(r+30)*math.cos(angle)
            ly = cy+(r+30)*math.sin(angle)
            anchor = "middle" if abs(math.cos(angle))<0.3 else ("start" if math.cos(angle)>0 else "end")
            svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" dominant-baseline="middle" font-size="11" fill="#7ba7c4" font-family="Inter,sans-serif">{label}</text>')

        # Player polygons
        for pname, vals, color in players_data:
            pts = []
            for i,v in enumerate(vals):
                angle = (i/N)*2*math.pi - math.pi/2
                rv = float(v)*r
                pts.append(f"{cx+rv*math.cos(angle):.1f},{cy+rv*math.sin(angle):.1f}")
            pts_str = " ".join(pts)
            ri = int(color[1:3],16)
            gi = int(color[3:5],16)
            bi = int(color[5:7],16)
            svg.append(f'<polygon points="{pts_str}" fill="rgba({ri},{gi},{bi},0.12)" stroke="{color}" stroke-width="2.5"/>')
            for i,v in enumerate(vals):
                angle = (i/N)*2*math.pi - math.pi/2
                rv = float(v)*r
                svg.append(f'<circle cx="{cx+rv*math.cos(angle):.1f}" cy="{cy+rv*math.sin(angle):.1f}" r="4" fill="{color}"/>')

        # Legend
        for i,(pname,_,color) in enumerate(players_data):
            lx = 40 + i*145
            short = pname[:15]+"…" if len(pname)>15 else pname
            svg.append(f'<rect x="{lx}" y="525" width="12" height="12" fill="{color}" rx="2"/>')
            svg.append(f'<text x="{lx+16}" y="535" font-size="10" fill="#c8e6f5" font-family="Inter,sans-serif">{short}</text>')

        svg.append("</svg>")
        components.html(f'<div style="background:#080f1a;padding:1rem;border-radius:12px;">{"".join(svg)}</div>', height=600)

        ri1,ri2 = st.columns(2)
        ri1.metric("Match Impact Score", f'{float(prow.get("match_impact_score",0)):.3f}')
        ri2.metric("Total Risk Score",   f'{float(prow.get("total_risk",0)):.3f}', delta="Lower is better", delta_color="inverse")

    # ── SIMILARITY SEARCH ─────────────────────────────────────────────────
    cric_divider()
    section("Similarity Search", "🔍")
    st.caption("Finds stylistically closest players using phase profile + performance fingerprint.")

    col_sel, col_k = st.columns([3,1])
    all_players = df["player"].tolist()
    target    = col_sel.selectbox("Select a player", all_players, index=0)
    top_k_sim = col_k.number_input("Top K", min_value=5, max_value=25, value=10, step=1)

    sim = get_similar_players(df, target, top_k=int(top_k_sim))

    if len(sim):
        trow = df[df["player"]==target].iloc[0]
        st.markdown(f"""
        <div class="player-card">
            <div class="pname">🎯 {target} {role_badge(trow['role'])}</div>
            <div class="pstat">
                Runs: <b>{int(trow.get('runs',0))}</b> &nbsp;|&nbsp;
                SR: <b>{trow.get('strike_rate',0):.1f}</b> &nbsp;|&nbsp;
                Wkts: <b>{int(trow.get('wickets',0))}</b> &nbsp;|&nbsp;
                Eco: <b>{trow.get('economy',0):.2f}</b> &nbsp;|&nbsp;
                Impact: <b>{trow.get('match_impact_score',0):.3f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sim_cols = [c for c in ["player","role","batting_role","bowling_role",
                                  "similarity","match_impact_score","total_risk",
                                  "runs","strike_rate","wickets","economy",
                                  "scouting_grade","analyst_recommendation"] if c in sim.columns]
        sim_show = sim[sim_cols].rename(columns={"similarity":"Similarity ↓"})
        st.dataframe(
            sim_show.style.format(
                {"Similarity ↓":"{:.3f}","match_impact_score":"{:.3f}",
                 "total_risk":"{:.3f}","strike_rate":"{:.1f}","economy":"{:.2f}"}
            ),
            use_container_width=True, height=380
        )
    else:
        st.warning("No similar players found.")

    # ── GAP-FILL RECOMMENDER ──────────────────────────────────────────────
    cric_divider()
    section("Gap-Fill Recommender", "🧩")
    st.caption("Select your current players in a role — get the best available replacements.")

    gap_options = ["Pacer (any)","Left-arm Pacer","Right-arm Pacer",
                   "Spinner (any)","Left-arm Spinner","Right-arm Spinner",
                   "Off-spinner","Leg-spinner","Opener","Top-4 (anchor)","Finisher"]

    g1, g2 = st.columns([2,1])
    gap_type    = g1.selectbox("Gap to fill", gap_options)
    allow_small = g2.checkbox("Allow 2-player unit", value=True)

    def gap_pool(d, gap):
        pools = {
            "Pacer (any)":       d[d["is_pacer"].astype(int)==1],
            "Left-arm Pacer":    d[d["is_left_arm_pacer"].astype(int)==1],
            "Right-arm Pacer":   d[d["is_right_arm_pacer"].astype(int)==1],
            "Spinner (any)":     d[d["is_spinner"].astype(int)==1],
            "Left-arm Spinner":  d[d["is_left_arm_spinner"].astype(int)==1],
            "Right-arm Spinner": d[d["is_right_arm_spinner"].astype(int)==1],
            "Off-spinner":       d[d["is_off_spinner"].astype(int)==1],
            "Leg-spinner":       d[d["is_leg_spinner"].astype(int)==1],
            "Opener":            d[d["is_opener"].astype(int)==1],
            "Top-4 (anchor)":    d[d["is_top4_candidate"].astype(int)==1],
            "Finisher":          d[d["is_finisher"].astype(int)==1],
        }
        return pools.get(gap, d)

    pool_df = gap_pool(df, gap_type)

    if len(pool_df)==0:
        st.warning("No players found for this gap type.")
        st.stop()

    current_players = st.multiselect(
        f"Your current players ({gap_type})",
        options=pool_df["player"].tolist(), default=[]
    )

    n_sel = len(current_players)
    if n_sel==0:
        st.info("Select at least 1 player to get recommendations.")
        st.stop()
    elif n_sel==1:
        st.info("**Mode: Player-style match** — finding closest stylistic alternatives.")
    elif n_sel==2 and not allow_small:
        st.warning("Select 3+ for stable unit match, or enable 'Allow 2-player unit'.")
        st.stop()
    elif n_sel==2:
        st.warning("**Mode: Small unit (2 players)** — less stable.")
    else:
        st.success(f"**Mode: Unit match ({n_sel} players)** — high confidence recommendations.")

    def feat_cols_for_gap(gap):
        bowl = ["pp_bowl_score","mid_bowl_score","death_bowl_score",
                "pp_eco","middle_eco","death_eco","pp_wkts","death_wkts",
                "economy","dot_ball_pct","match_impact_score","total_risk"]
        bat  = ["pp_bat_score","mid_bat_score","death_bat_score",
                "pp_sr","middle_sr","strike_rate","pp_runs","runs",
                "boundary_pct","match_impact_score","total_risk"]
        death= ["death_bat_score","death_sr","death_runs","strike_rate",
                "boundary_pct","match_impact_score","total_risk"]
        if gap in ["Pacer (any)","Left-arm Pacer","Right-arm Pacer",
                   "Spinner (any)","Left-arm Spinner","Right-arm Spinner","Off-spinner","Leg-spinner"]:
            return [c for c in bowl if c in df.columns]
        if gap in ["Opener","Top-4 (anchor)"]:
            return [c for c in bat if c in df.columns]
        if gap=="Finisher":
            return [c for c in death if c in df.columns]
        return [c for c in ["match_impact_score","total_risk","strike_rate","economy"] if c in df.columns]

    feat_cols  = feat_cols_for_gap(gap_type)
    base       = df[df["player"].isin(current_players)].copy()
    cand       = pool_df[~pool_df["player"].isin(current_players)].copy()

    if len(cand)==0:
        st.warning("No candidates available.")
        st.stop()

    scaler    = StandardScaler()
    all_feats = pd.concat([base[feat_cols],cand[feat_cols]]).apply(pd.to_numeric,errors="coerce").fillna(0.0)
    scaled    = scaler.fit_transform(all_feats.values)
    base_s    = scaled[:len(base)]
    cand_s    = scaled[len(base):]

    centroid  = base_s.mean(axis=0).reshape(1,-1)
    sims      = cosine_similarity(cand_s, centroid).reshape(-1)
    cand      = cand.copy()
    cand["unit_fit_score"] = sims
    cand["combined_reco"]  = (
        0.70*cand["unit_fit_score"]
        + 0.30*norm01(cand["match_impact_score"])
        - 0.20*norm01(cand["total_risk"])
    )

    rec      = cand.sort_values("combined_reco",ascending=False).head(10)
    unit_avg = base[feat_cols].apply(pd.to_numeric,errors="coerce").fillna(0.0).mean()

    col_up, col_rec = st.columns([1,2])
    with col_up:
        section("Unit Profile", "📐")
        unit_df = unit_avg.reset_index()
        unit_df.columns = ["Metric","Unit Avg"]
        unit_df["Unit Avg"] = unit_df["Unit Avg"].round(3)
        st.dataframe(unit_df, use_container_width=True, height=300)

    with col_rec:
        section("Top 10 Recommendations", "⭐")
        show_rec = [c for c in ["player","role","age","bat_hand","bowling_arm",
                                  "batting_role","bowling_role","unit_fit_score",
                                  "combined_reco","match_impact_score","total_risk",
                                  "scouting_grade","analyst_recommendation"] if c in rec.columns]
        st.dataframe(
            rec[show_rec].style.format(
                {"unit_fit_score":"{:.3f}","combined_reco":"{:.3f}",
                 "match_impact_score":"{:.3f}","total_risk":"{:.3f}"}
            ),
            use_container_width=True, height=300
        )

    # Explainability
    cric_divider()
    section("Explainability — Top 5", "🔬")
    for _, r in rec.head(5).iterrows():
        diffs = {}
        for c in feat_cols:
            try: diffs[c] = float(r[c]) - float(unit_avg.get(c,0.0))
            except: continue
        ranked = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)
        best   = ranked[:3]
        worst  = ranked[-1:] if ranked else []

        st.markdown(f"""
        <div class="explain-card">
            <div class="ename">{r['player']} {role_badge(r['role'])}</div>
            <div class="escores">combined_reco: <b>{r.get('combined_reco',0):.3f}</b> &nbsp;|&nbsp; unit_fit: <b>{r.get('unit_fit_score',0):.3f}</b> &nbsp;|&nbsp; impact: <b>{r.get('match_impact_score',0):.3f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        for k, v in best:
            st.markdown(f'<span class="explain-pos">▲ <b>{k}</b> better than unit avg ({v:+.3f})</span>', unsafe_allow_html=True)
        for k, v in worst:
            st.markdown(f'<span class="explain-neg">▼ <b>{k}</b> trade-off vs unit ({v:+.3f})</span>', unsafe_allow_html=True)
        st.write("")

    # Downloads
    cric_divider()
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Full Scout Table",    to_csv_bytes(df),       "scout_full.csv",    "text/csv")
    d2.download_button("⬇ Filtered Players",    to_csv_bytes(filtered), "scout_filtered.csv","text/csv")
    d3.download_button("⬇ Recommendations",     to_csv_bytes(rec),      "gap_fill_reco.csv", "text/csv")


# ═══════════════════════════════════════════════════════
# AUCTION MODE
# ═══════════════════════════════════════════════════════
def run_auction_mode():
    # Read from unified session state
    df_base   = st.session_state["df_master"].copy()
    players   = st.session_state.get("players_raw", pd.DataFrame())
    perf      = st.session_state.get("perf_raw",    pd.DataFrame())

    # Check for contracts and budget
    if "contracts_raw" not in st.session_state or "budget_raw" not in st.session_state:
        st.markdown("""
        <div class="mapper-card">
            <div class="mc-title">💰 Auction Mode needs 2 more files</div>
            <div class="mc-sub">
                Your player data is already loaded. To use Auction Mode, also upload:<br><br>
                <b>Contracts / Salary CSV:</b> player IDs + salary/wage/price column<br>
                <b>Budget CSV:</b> total budget + squad size constraints
            </div>
        </div>
        """, unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        with ac1:
            cf = st.file_uploader("Contracts / Salary CSV", type="csv", key="au_ct_late")
        with ac2:
            bf = st.file_uploader("Budget CSV", type="csv", key="au_bd_late")
        if cf:
            st.session_state["contracts_raw"] = pd.read_csv(cf)
        if bf:
            st.session_state["budget_raw"] = pd.read_csv(bf)
        if "contracts_raw" not in st.session_state or "budget_raw" not in st.session_state:
            st.info("Upload both files above to continue.")
            st.stop()

    contracts = st.session_state["contracts_raw"].copy()
    budget_df = st.session_state["budget_raw"].copy()

    # ── Auto-detect salary column ─────────────────────────────────────────
    c_detected = auto_detect_columns(contracts)
    salary_col = c_detected.get("current_salary_lakh", (None,0))[0]

    if salary_col is None:
        st.error("❌ Could not detect a salary/wage column in contracts CSV. Please ensure it has a column like 'salary', 'wage', 'base_price', or 'current_salary_lakh'.")
        st.stop()

    if salary_col != "current_salary_lakh":
        contracts = contracts.rename(columns={salary_col: "current_salary_lakh"})

    contracts["current_salary_lakh"] = pd.to_numeric(contracts["current_salary_lakh"],errors="coerce").fillna(0.0)

    # ── Auto-detect budget row ────────────────────────────────────────────
    b_detected  = auto_detect_columns(budget_df)
    budget_col  = b_detected.get("budget_lakh",  (None,0))[0]
    maxpl_col   = b_detected.get("max_players",  (None,0))[0]
    minbat_col  = b_detected.get("min_bat",       (None,0))[0]
    minbowl_col = b_detected.get("min_bowl",      (None,0))[0]
    minar_col   = b_detected.get("min_ar",        (None,0))[0]
    minwk_col   = b_detected.get("min_wk",        (None,0))[0]

    def get_budget_val(col, default):
        if col and col in budget_df.columns:
            return budget_df[col].iloc[0]
        return default

    default_budget = get_budget_val(budget_col,  10000)
    default_maxpl  = get_budget_val(maxpl_col,   25)
    default_minbat = get_budget_val(minbat_col,  8)
    default_minbowl= get_budget_val(minbowl_col, 8)
    default_minar  = get_budget_val(minar_col,   4)
    default_minwk  = get_budget_val(minwk_col,   2)

    # Show detection
    cric_divider()
    section("Column Detection", "🧠")
    all_det = {**auto_detect_columns(players), **auto_detect_columns(perf), **c_detected, **b_detected}
    req = ["player","player_id","role","matches","runs","strike_rate",
           "wickets","economy","current_salary_lakh","budget_lakh"]
    render_mapping_summary(all_det, req)

    # Merge contracts into master df
    with st.spinner("Preparing auction data..."):
        df = build_base_df(players, perf, contracts=contracts)
        df["current_salary_lakh"] = pd.to_numeric(df.get("current_salary_lakh",0),errors="coerce").fillna(0.0)

    # ── CONTEXT ───────────────────────────────────────────────────────────
    cric_divider()
    section("Match Context", "🏟️")
    cc1,cc2,cc3,cc4 = st.columns(4)
    pitch_type  = cc1.selectbox("Home pitch",      ["Flat/True","Spin-friendly","Pace/Bounce"])
    season_goal = cc2.selectbox("Season strategy", ["Balanced","Batting dominance","Bowling dominance"])
    risk_pref   = cc3.selectbox("Risk preference", ["Balanced","Risk-averse","High-upside"])
    auction_mode= cc4.checkbox("Auction pricing",  value=True)

    cric_divider()
    section("Opponent Profile", "🎯")
    opp_profiles = {
        "Balanced":               {"spin":0.5,"pace":0.5,"death":0.5,"pp":0.5},
        "Spin choke team":        {"spin":0.85,"pace":0.35,"death":0.55,"pp":0.45},
        "Pace/bounce attack":     {"spin":0.35,"pace":0.85,"death":0.65,"pp":0.55},
        "Death-over specialists": {"spin":0.45,"pace":0.55,"death":0.90,"pp":0.45},
        "Powerplay smashers":     {"spin":0.45,"pace":0.55,"death":0.55,"pp":0.90},
    }
    oc1,oc2 = st.columns([2,1])
    opp_sel  = oc1.selectbox("Opponent type", list(opp_profiles.keys()))
    override = oc2.checkbox("Manual sliders", value=False)
    base_opp = opp_profiles[opp_sel]
    o1,o2,o3,o4 = st.columns(4)
    opp_spin  = o1.slider("Spin threat",        0.0,1.0,float(base_opp["spin"]),  0.05,disabled=not override)
    opp_pace  = o2.slider("Pace threat",         0.0,1.0,float(base_opp["pace"]),  0.05,disabled=not override)
    opp_death = o3.slider("Death bowl strength", 0.0,1.0,float(base_opp["death"]), 0.05,disabled=not override)
    opp_pp    = o4.slider("PP aggressiveness",   0.0,1.0,float(base_opp["pp"]),    0.05,disabled=not override)

    cric_divider()
    section("Auction Settings", "💰")
    a1,a2,a3 = st.columns(3)
    inflation     = a1.slider("Inflation multiplier", 1.0,2.5,1.35,0.05,disabled=not auction_mode)
    reserve_floor = a2.number_input("Reserve floor (lakh)", value=120.0, step=10.0, disabled=not auction_mode)
    budget_lakh   = a3.number_input("Budget (lakh)", value=float(default_budget), step=100.0)

    cric_divider()
    section("Squad Constraints", "⚙️")
    c5,c6,c7,c8 = st.columns(4)
    max_players = c5.number_input("Max squad",  value=int(default_maxpl),   step=1)
    min_bat     = c6.number_input("Min BAT",    value=int(default_minbat),  step=1)
    min_bowl    = c7.number_input("Min BOWL",   value=int(default_minbowl), step=1)
    min_ar      = c8.number_input("Min AR",     value=int(default_minar),   step=1)
    c9,c10 = st.columns(2)
    min_wk             = c9.number_input("Min WK",        value=int(default_minwk), step=1)
    max_overseas_squad = c10.number_input("Max Overseas", value=8, step=1)
    min_role = {"BAT":min_bat,"BOWL":min_bowl,"AR":min_ar,"WK":min_wk}

    cric_divider()
    section("Balance Constraints", "⚖️")
    b1,b2,b3,b4 = st.columns(4)
    min_spinners  = b1.number_input("Min Spinners",      value=3,step=1)
    min_pacers    = b2.number_input("Min Pacers",        value=3,step=1)
    min_death_bowl= b3.number_input("Min Death Bowlers", value=2,step=1)
    min_death_hit = b4.number_input("Min Death Hitters", value=2,step=1)
    b5,b6,b7 = st.columns(3)
    min_pp_bowl  = b5.number_input("Min PP Bowlers", value=2,step=1)
    min_openers  = b6.number_input("Min Openers",    value=2,step=1)
    min_finishers= b7.number_input("Min Finishers",  value=2,step=1)
    enforce_left = st.checkbox("Enforce left-right balance (≥1 lefty in top 4)", value=True)

    cric_divider()
    section("Retentions / RTM", "🔒")
    r1,r2 = st.columns([2,1])
    retained_players = r1.multiselect("Retained/locked players", df["player"].tolist(), default=[])
    default_ret_cost = r2.number_input("Default retained cost (lakh)", value=600.0, step=50.0)

    retained_costs = {}
    if retained_players:
        rc_cols = st.columns(min(len(retained_players),4))
        for i,p in enumerate(retained_players[:20]):
            retained_costs[p] = rc_cols[i%4].number_input(p, value=float(default_ret_cost), step=50.0, key=f"rc_{p}")

    locked_set       = set(retained_players)
    retained_total   = sum(retained_costs.get(p,default_ret_cost) for p in retained_players)
    budget_after_ret = float(budget_lakh - retained_total)
    st.metric("Budget after retentions", f"₹ {budget_after_ret:,.1f} lakh",
              delta=f"-₹{retained_total:,.1f} retained" if retained_total>0 else None)
    if budget_after_ret<=0:
        st.error("❌ Retentions exceed budget.")
        st.stop()

    # ── Pricing ───────────────────────────────────────────────────────────
    df["auction_price_lakh"] = df["current_salary_lakh"].copy()
    if auction_mode:
        df["auction_price_lakh"] = (df["current_salary_lakh"]*float(inflation)).clip(lower=float(reserve_floor))
    price_col = "auction_price_lakh" if auction_mode else "current_salary_lakh"
    df["price_used_lakh"] = df[price_col].copy()
    if retained_players:
        df["price_used_lakh"] = df["price_used_lakh"].astype(float)
        for p in retained_players:
            df.loc[df["player"]==p,"price_used_lakh"] = float(retained_costs.get(p,default_ret_cost))
    price_col = "price_used_lakh"

    rp_map = {"Risk-averse":1.2,"High-upside":0.4,"Balanced":0.8}
    df["risk_adjustment"] = (1-rp_map[risk_pref]*df["total_risk"]).clip(lower=0.25,upper=1.05)

    fit = np.ones(len(df),dtype=float)
    if pitch_type=="Spin-friendly":
        fit += 0.12*df["is_spinner"].astype(float)+0.05*(df["batting_role"]=="ANCHOR").astype(float)
        fit -= 0.04*df["is_pacer"].astype(float)
    elif pitch_type=="Pace/Bounce":
        fit += 0.12*df["is_pacer"].astype(float)+0.05*norm01(df["pp_sr"])
        fit -= 0.04*df["is_spinner"].astype(float)
    else:
        fit += 0.10*norm01(df["strike_rate"])+0.08*(df["batting_role"]=="FINISHER").astype(float)
    df["pitch_fit"] = np.clip(fit,0.80,1.30)

    cric_divider()
    section("Phase Importance", "⏱️")
    p1,p2,p3 = st.columns(3)
    w_pp    = p1.slider("Powerplay weight",    0.0,2.0,1.0,0.1)
    w_mid   = p2.slider("Middle overs weight", 0.0,2.0,1.0,0.1)
    w_death = p3.slider("Death overs weight",  0.0,2.0,1.2,0.1)
    df = compute_phase_scores(df, w_pp, w_mid, w_death)

    lefty   = (df["bat_hand"].astype(str)=="L").astype(float)
    anchor  = (df["batting_role"]=="ANCHOR").astype(float)
    opener  = (df["batting_role"]=="OPENER").astype(float)
    pp_bowl = (df["bowling_role"]=="PP").astype(float)
    pacer   = df["is_pacer"].astype(float)
    spinner = df["is_spinner"].astype(float)

    opp_mult = np.ones(len(df),dtype=float)
    opp_mult += 0.10*opp_spin *(0.9*lefty +0.6*anchor+0.6*spinner+0.2*df["mid_bat_score"])
    opp_mult += 0.10*opp_pace *(0.9*pacer +0.6*opener+0.5*pp_bowl+0.2*df["pp_bat_score"])
    opp_mult += 0.08*opp_death*(0.9*anchor+0.4*df["mid_bat_score"])
    opp_mult += 0.08*opp_pp   *(0.9*pp_bowl+0.5*(1-norm01(df["pp_eco"])))
    df["opponent_fit"] = np.clip(opp_mult,0.85,1.22)

    feat_pool = ["role","age","runs","wickets","matches","strike_rate","economy",
                 "dot_ball_pct","boundary_pct","match_impact_score"]
    X = df[[c for c in feat_pool if c in df.columns]].copy()
    y = df["current_salary_lakh"].copy()
    cat_cols = ["role"] if "role" in X.columns else []
    ct    = ColumnTransformer([("ohe",OneHotEncoder(handle_unknown="ignore"),cat_cols)],remainder="passthrough")
    model = Pipeline([("prep",ct),("reg",Ridge(alpha=1.0))])
    model.fit(X, y)
    df["fair_salary_lakh"] = np.clip(model.predict(X),0,None)
    df["value_gap"] = df["fair_salary_lakh"] - df["price_used_lakh"]

    df["phase_versatility"] = (
        0.5*((df["is_pp_batter"].astype(int)==1)|(df["is_pp_bowler"].astype(int)==1)).astype(float)+
        0.5*((df["is_death_hitter"].astype(int)==1)|(df["is_death_bowler"].astype(int)==1)).astype(float)
    )
    df["flex_score"] = norm01(0.55*(df["role"].isin(["AR","WK"]).astype(float))+0.45*norm01(df["phase_versatility"]))

    cric_divider()
    section("Auction Strategy & Objective Weights", "🎚️")
    s1,s2,s3 = st.columns(3)
    strategy           = s1.selectbox("Squad style", ["Balanced","Star-heavy","Depth-heavy"])
    max_single_pct     = s2.slider("Max single-player spend (%)",10,60,30,1)
    price_conc_penalty = s3.slider("Price concentration penalty",0.0,5.0,1.0,0.1)
    max_single_price   = float(budget_after_ret)*(float(max_single_pct)/100.0)

    bw = {"value":1.0,"impact":1.2,"fit":1.0,"opp":1.0,"flex":0.8,"risk":1.0}
    if strategy=="Star-heavy":   bw.update({"value":0.8,"impact":1.35,"fit":1.15,"opp":1.15})
    elif strategy=="Depth-heavy":bw.update({"value":1.25,"impact":1.10})

    mw1,mw2,mw3,mw4,mw5,mw6 = st.columns(6)
    w_value  = mw1.slider("Value gap",    0.0,2.0,bw["value"],  0.05)
    w_impact = mw2.slider("Impact",       0.0,2.0,bw["impact"], 0.05)
    w_fit    = mw3.slider("Pitch fit",    0.0,2.0,bw["fit"],    0.05)
    w_opp    = mw4.slider("Opponent fit", 0.0,2.0,bw["opp"],    0.05)
    w_flex   = mw5.slider("Flex",         0.0,2.0,bw["flex"],   0.05)
    w_risk   = mw6.slider("Risk penalty", 0.0,2.0,bw["risk"],   0.05)

    df["value_gap_norm"] = norm01(df["value_gap"])
    df["impact_norm"]    = norm01(df["match_impact_score"])
    df["fit_norm"]       = norm01(df["pitch_fit"])
    df["opp_norm"]       = norm01(df["opponent_fit"])
    df["risk_norm"]      = norm01(df["total_risk"])

    df["objective_score"] = (
        w_value*df["value_gap_norm"] + w_impact*df["impact_norm"] +
        w_fit*df["fit_norm"] + w_opp*df["opp_norm"] +
        w_flex*df["flex_score"] - w_risk*df["risk_norm"]
    ) * df["risk_adjustment"]

    df["xi_score"] = (
        0.28*df["impact_norm"]+0.20*df["fit_norm"]+0.22*df["opp_norm"]+
        0.15*df["flex_score"] +0.15*df["value_gap_norm"]
    ) * df["risk_adjustment"]

    cric_divider()
    section("Top Player Rankings", "📊")
    top_show = [c for c in ["player","role","age","bat_hand","batting_role","bowling_role",
                              "is_spinner","is_pacer","is_overseas","price_used_lakh",
                              "fair_salary_lakh","value_gap","match_impact_score",
                              "pitch_fit","opponent_fit","flex_score","total_risk",
                              "objective_score"] if c in df.columns]
    st.dataframe(
        df[top_show].sort_values("objective_score",ascending=False).head(50).style.format(
            {"objective_score":"{:.3f}","value_gap":"{:.1f}","match_impact_score":"{:.3f}","total_risk":"{:.3f}"}
        ),
        use_container_width=True, height=420
    )

    cric_divider()
    section("Optimised Squad Selection", "🤖")
    extra_min_flags = {
        "is_spinner":int(min_spinners),"is_pacer":int(min_pacers),
        "is_death_bowler2":int(min_death_bowl),"is_death_hitter":int(min_death_hit),
        "is_pp_bowler2":int(min_pp_bowl),"is_opener":int(min_openers),"is_finisher":int(min_finishers),
    }
    extra_max_flags = {"is_overseas":int(max_overseas_squad)}

    soft_mode      = st.checkbox("Soft constraints", value=True)
    penalty_weight = st.slider("Soft penalty", 0.5,10.0,2.5,0.1,disabled=not soft_mode)

    with st.spinner("Running squad optimiser..."):
        squad, sm = optimize_squad_soft(
            df=df, budget_limit=float(budget_after_ret), max_players=int(max_players),
            min_role=min_role, price_col=price_col,
            extra_min_flags=extra_min_flags, extra_max_flags=extra_max_flags,
            lock_players=locked_set if locked_set else None,
            max_single_price=max_single_price, penalty_weight=float(penalty_weight),
            price_concentration_penalty=float(price_conc_penalty),
        )

    s1,s2,s3,s4,s5,s6 = st.columns(6)
    s1.metric("Players",     sm.get("count",0))
    s2.metric("Spend",       f"{sm.get('spend',0):,.1f}")
    s3.metric("Objective",   f"{sm.get('objective',0):.2f}")
    s4.metric("BAT",         sm.get("BAT",0))
    s5.metric("BOWL",        sm.get("BOWL",0))
    s6.metric("AR+WK",       sm.get("AR",0)+sm.get("WK",0))

    if sm.get("violations"):
        v = {k:round(v,2) for k,v in sm["violations"].items() if v and v>0.001}
        if v: st.warning(f"⚠️ Soft shortfalls: {v}")

    if len(squad)==0:
        st.error("❌ No squad returned.")
        st.stop()

    squad_show = [c for c in ["player","role","bat_hand","batting_role","bowling_role",
                               "is_spinner","is_pacer","is_overseas","price_used_lakh",
                               "fair_salary_lakh","value_gap","match_impact_score",
                               "pitch_fit","opponent_fit","flex_score","total_risk",
                               "objective_score"] if c in squad.columns]
    st.dataframe(
        squad[squad_show].sort_values("objective_score",ascending=False).style.format(
            {"objective_score":"{:.3f}","value_gap":"{:.1f}","total_risk":"{:.3f}"}
        ),
        use_container_width=True, height=420
    )

    cric_divider()
    section("Best Playing XI", "🏏")
    x1,x2,x3,x4 = st.columns(4)
    xi_size      = x1.number_input("XI size",    value=11,step=1)
    xi_min_bat   = x2.number_input("XI Min BAT", value=4, step=1)
    xi_min_bowl  = x3.number_input("XI Min BOWL",value=4, step=1)
    xi_min_wk    = x4.number_input("XI Min WK",  value=1, step=1)
    x5,x6 = st.columns(2)
    xi_min_ar       = x5.number_input("XI Min AR",       value=1,step=1)
    max_overseas_xi = x6.number_input("XI Max Overseas", value=4,step=1)

    with st.spinner("Picking best XI..."):
        xi, xm = pick_best_xi(squad=squad, xi_size=int(xi_size),
                               xi_min_role={"BAT":xi_min_bat,"BOWL":xi_min_bowl,"AR":xi_min_ar,"WK":xi_min_wk},
                               max_overseas_xi=int(max_overseas_xi), enforce_left_in_top4=enforce_left)

    if len(xi)==0:
        st.warning("⚠️ No feasible XI — relax constraints.")
    else:
        st.metric("XI total score", f"{xm.get('xi_score',0):.3f}")
        xi_show = [c for c in ["player","role","bat_hand","batting_role","bowling_role",
                                 "is_spinner","is_pacer","is_overseas","match_impact_score",
                                 "pitch_fit","opponent_fit","flex_score","total_risk","xi_score"] if c in xi.columns]
        st.dataframe(
            xi[xi_show].sort_values("xi_score",ascending=False).style.format(
                {"xi_score":"{:.3f}","match_impact_score":"{:.3f}","total_risk":"{:.3f}"}
            ),
            use_container_width=True, height=420
        )

    cric_divider()
    d1,d2,d3 = st.columns(3)
    d1.download_button("⬇ Full Table", to_csv_bytes(df),    "auction_full.csv",  "text/csv")
    d2.download_button("⬇ Squad",      to_csv_bytes(squad), "auction_squad.csv", "text/csv")
    d3.download_button("⬇ Best XI",    to_csv_bytes(xi) if len(xi) else b"", "auction_xi.csv","text/csv")


# ═══════════════════════════════════════════════════════
# HIGHLIGHTS GENERATOR
# ═══════════════════════════════════════════════════════
def run_highlights_mode():
    section("Upload Source Video", "🎬")

    if "hl_video_path" not in st.session_state:
        st.session_state["hl_video_path"] = None

    tab_file, tab_yt = st.tabs(["📂 Upload Video","▶️ YouTube Link"])

    with tab_file:
        vid = st.file_uploader("Upload video (mp4/mov/mkv)", type=["mp4","mov","mkv"], key="hl_upload")
        if vid:
            tmp_dir = tempfile.mkdtemp()
            vpath   = os.path.join(tmp_dir, vid.name)
            with open(vpath,"wb") as f: f.write(vid.getbuffer())
            st.session_state["hl_video_path"] = vpath
            st.success("✅ Video uploaded")
            st.video(vpath)

    with tab_yt:
        yt_url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...", key="hl_yt")
        st.caption("Requires `yt-dlp` + `ffmpeg` on the server.")
        if yt_url and st.button("Fetch from YouTube", key="hl_fetch"):
            tmp_dir = tempfile.mkdtemp()
            out_tpl = os.path.join(tmp_dir,"input.%(ext)s")
            try:
                subprocess.check_call(["yt-dlp","-f","bv*+ba/b","-o",out_tpl,yt_url])
                vpath = next((os.path.join(tmp_dir,f) for f in os.listdir(tmp_dir) if f.startswith("input.")),None)
                if vpath and os.path.exists(vpath):
                    st.session_state["hl_video_path"] = vpath
                    st.success("✅ Downloaded")
                    st.video(vpath)
                else:
                    st.error("File not found after download.")
            except Exception as e:
                st.error(f"Download failed: {e}")

    cric_divider()
    section("Generate Highlights Clip", "✂️")
    c1,c2,c3 = st.columns(3)
    start_sec = c1.number_input("Start time (seconds)", min_value=0, value=0, step=5)
    clip_secs = c2.number_input("Clip length (seconds)", min_value=5, max_value=600, value=60, step=5)
    out_name  = c3.text_input("Output filename", value="highlights.mp4")

    if st.button("⚡ Generate Highlights", type="primary"):
        vpath = st.session_state.get("hl_video_path")
        if not vpath or not os.path.exists(vpath):
            st.error("Upload a video first.")
            return
        tmp_out  = tempfile.mkdtemp()
        out_path = os.path.join(tmp_out, out_name)
        cmd = ["ffmpeg","-y","-i",vpath,"-ss",str(int(start_sec)),"-t",str(int(clip_secs)),"-c","copy",out_path]
        try:
            with st.spinner("Processing..."): subprocess.check_call(cmd)
            st.success("✅ Done")
            st.video(out_path)
            with open(out_path,"rb") as f:
                st.download_button("⬇ Download Highlights", data=f.read(), file_name=out_name, mime="video/mp4")
        except Exception as e:
            st.error(f"❌ ffmpeg failed: {e}")


# ═══════════════════════════════════════════════════════
# SIDEBAR + ROUTING
# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# CUSTOM INTELLIGENCE MODE
# ═══════════════════════════════════════════════════════
def run_custom_intelligence():
    # Read from unified session state — use raw files for maximum column access
    players = st.session_state.get("players_raw", pd.DataFrame())
    perf    = st.session_state.get("perf_raw",    pd.DataFrame())

    # Merge raw files to get ALL columns available
    df = smart_merge(players, perf)
    df = apply_column_mapping(df, {f: col for f, (col,_) in auto_detect_columns(df).items()})
    if "player" not in df.columns:
        name_candidates = [c for c in df.columns if "name" in c.lower()]
        df["player"] = df[name_candidates[0]] if name_candidates else ["Player_"+str(i) for i in df.index]
    if "player_id" not in df.columns:
        df["player_id"] = df.index + 1

    # All numeric columns available
    exclude = ["player_id","is_spinner","is_pacer","is_overseas","injury_risk",
               "is_pp_bowler","is_death_bowler","is_pp_batter","is_death_hitter"]
    available_metrics = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    # ── STEP 1: SELECT METRICS ────────────────────────────────────────────
    cric_divider()
    section("Step 1 — Select Your Metrics", "📊")
    st.caption(f"Choose any metrics from your data. {len(available_metrics)} metrics available.")

    selected_metrics = st.multiselect(
        "Choose metrics to include in your analysis",
        options=available_metrics,
        default=[],
        placeholder="Select metrics — choose as many as you want...",
        key="ci_metrics"
    )

    if not selected_metrics:
        st.info("👆 Select at least 2 metrics to continue.")
        st.stop()

    st.success(f"✅ {len(selected_metrics)} metrics selected")

    # ── STEP 2: PRIORITY WEIGHTING (OPTIONAL) ────────────────────────────
    cric_divider()
    section("Step 2 — Priority Weighting (Optional)", "⚖️")

    use_weights = st.checkbox(
        "I want to set priority weights for specific metrics",
        value=False,
        key="ci_use_weights"
    )

    if not use_weights:
        st.info(f"All {len(selected_metrics)} metrics carry equal weight. Toggle above to customise.")
        # Equal weights
        weights = {m: 1.0/len(selected_metrics) for m in selected_metrics}

    else:
        st.caption("Pick up to 5 priority metrics and set their importance. All other metrics share the remaining weight equally.")

        # Pick priority metrics
        priority_metrics = st.multiselect(
            "Select up to 5 priority metrics",
            options=selected_metrics,
            default=selected_metrics[:3] if len(selected_metrics) >= 3 else selected_metrics,
            max_selections=5,
            key="ci_priority"
        )

        if not priority_metrics:
            st.warning("Select at least 1 priority metric or disable priority weighting.")
            weights = {m: 1.0/len(selected_metrics) for m in selected_metrics}
        else:
            # Sliders for priority metrics
            st.markdown("**Set importance for your priority metrics:**")
            cols = st.columns(min(len(priority_metrics), 5))
            priority_weights = {}
            for i, metric in enumerate(priority_metrics):
                default_w = min(30, int(80/len(priority_metrics)))
                priority_weights[metric] = cols[i].slider(
                    metric.replace("_"," ").title(),
                    min_value=5, max_value=80,
                    value=default_w, step=5,
                    key=f"pw_{metric}"
                )

            total_priority = sum(priority_weights.values())

            if total_priority >= 100:
                st.error(f"❌ Priority weights total {total_priority}% — must be less than 100% to leave room for other metrics.")
                weights = {m: 1.0/len(selected_metrics) for m in selected_metrics}
            else:
                remaining_pct = 100 - total_priority
                non_priority = [m for m in selected_metrics if m not in priority_metrics]
                equal_share = remaining_pct / len(non_priority) if non_priority else 0

                # Display summary
                w1, w2, w3 = st.columns(3)
                w1.metric("Priority metrics weight", f"{total_priority}%")
                w2.metric("Other metrics weight", f"{remaining_pct}%")
                w3.metric("Each other metric gets", f"{equal_share:.1f}%")

                st.success(f"✅ {len(priority_metrics)} priority metrics set · {len(non_priority)} metrics share remaining {remaining_pct}% equally")

                # Build final weights (normalised to sum=1)
                weights = {}
                for m in priority_metrics:
                    weights[m] = priority_weights[m] / 100.0
                for m in non_priority:
                    weights[m] = equal_share / 100.0

    # ── COMPUTE CUSTOM SCORE ──────────────────────────────────────────────
    df_clean = df.copy()
    for m in selected_metrics:
        if m in df_clean.columns:
            df_clean[m] = pd.to_numeric(df_clean[m], errors="coerce").fillna(0)

    df_clean["custom_score"] = sum(
        weights[m] * norm01(df_clean[m])
        for m in selected_metrics
        if m in df_clean.columns
    )

    # ── RESULTS ───────────────────────────────────────────────────────────
    cric_divider()
    section("Your Custom Analysis", "🏆")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Players",  len(df_clean))
    s2.metric("Metrics Used",   len(selected_metrics))
    s3.metric("Top Score",      f"{df_clean['custom_score'].max():.3f}")
    s4.metric("Avg Score",      f"{df_clean['custom_score'].mean():.3f}")

    show_cols = ["player"] + (["role"] if "role" in df_clean.columns else []) +                 (["age"] if "age" in df_clean.columns else []) +                 selected_metrics[:8] + ["custom_score"]
    show_cols = [c for c in dict.fromkeys(show_cols) if c in df_clean.columns]

    st.dataframe(
        df_clean[show_cols].sort_values("custom_score", ascending=False).head(50).style.format(
            {"custom_score": "{:.3f}"}
        ),
        use_container_width=True, height=440
    )

    # ── SIMILARITY SEARCH ─────────────────────────────────────────────────
    cric_divider()
    section("Similarity Search — Your Metrics", "🔍")
    st.caption("Find players most similar to any player based on YOUR chosen metrics.")

    target = st.selectbox("Select a player", df_clean["player"].tolist(), key="ci_sim_target")
    trow = df_clean[df_clean["player"]==target].head(1)

    if len(trow) > 0:
        feat_data = df_clean[selected_metrics].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feat_data)

        target_pos = df_clean[df_clean["player"]==target].index[0]
        df_positions = list(df_clean.index)
        t_pos_in_array = df_positions.index(target_pos)

        t_vec = scaled[t_pos_in_array].reshape(1,-1)
        cand_mask = [i for i in range(len(scaled)) if i != t_pos_in_array]
        cand_scaled = scaled[cand_mask]
        cand_df = df_clean.iloc[cand_mask].copy()

        sims = cosine_similarity(cand_scaled, t_vec).reshape(-1)
        cand_df["similarity"] = sims
        sim_show = cand_df.sort_values("similarity", ascending=False).head(10)

        st.markdown(f"""
        <div class="player-card">
            <div class="pname">🎯 {target}</div>
            <div class="pstat">Custom Score: <b>{float(trow["custom_score"].values[0]):.3f}</b> &nbsp;|&nbsp; Metrics: <b>{len(selected_metrics)}</b></div>
        </div>
        """, unsafe_allow_html=True)

        sim_cols = ["player"] + (["role"] if "role" in sim_show.columns else []) + ["similarity","custom_score"]
        st.dataframe(
            sim_show[sim_cols].style.format({"similarity":"{:.3f}","custom_score":"{:.3f}"}),
            use_container_width=True, height=380
        )

    # ── SHORTLIST ─────────────────────────────────────────────────────────
    cric_divider()
    section("Shortlist", "📋")

    if "ci_shortlist" not in st.session_state:
        st.session_state["ci_shortlist"] = []

    sl1, sl2 = st.columns([3,1])
    add_p = sl1.multiselect(
        "Add players to shortlist",
        options=[p for p in df_clean["player"].tolist() if p not in st.session_state["ci_shortlist"]],
        default=[], key="ci_shortlist_add"
    )
    if sl2.button("Add to Shortlist", type="primary", key="ci_add_btn"):
        for p in add_p:
            if p not in st.session_state["ci_shortlist"]:
                st.session_state["ci_shortlist"].append(p)
        st.rerun()

    if len(st.session_state["ci_shortlist"]) > 0:
        st.markdown(f"**{len(st.session_state['ci_shortlist'])} players shortlisted**")
        shortlist_df = df_clean[df_clean["player"].isin(st.session_state["ci_shortlist"])]
        sl_cols = ["player"] + selected_metrics[:6] + ["custom_score"]
        sl_cols = [c for c in sl_cols if c in shortlist_df.columns]
        st.dataframe(shortlist_df[sl_cols].style.format({"custom_score":"{:.3f}"}),
                    use_container_width=True)

        c1,c2,c3 = st.columns(3)
        rem = c1.selectbox("Remove", ["Select"]+st.session_state["ci_shortlist"], key="ci_rem")
        if c1.button("Remove", key="ci_rem_btn"):
            if rem != "Select":
                st.session_state["ci_shortlist"].remove(rem)
                st.rerun()
        if c2.button("Clear All", key="ci_clear"):
            st.session_state["ci_shortlist"] = []
            st.rerun()
        c3.download_button("⬇ Export Shortlist", to_csv_bytes(shortlist_df),
                          "custom_shortlist.csv", "text/csv")
    else:
        st.info("No players shortlisted yet.")

    # ── DOWNLOAD ──────────────────────────────────────────────────────────
    cric_divider()
    st.download_button("⬇ Download Full Custom Analysis", to_csv_bytes(df_clean[show_cols]),
                      "custom_analysis.csv", "text/csv")



# ═══════════════════════════════════════════════════════
# LOGIN SYSTEM
# ═══════════════════════════════════════════════════════
def check_login():
    """Password gate using Render environment variable."""
    
    # Get password from environment variable
    correct_password = os.environ.get("CRICINTEL_PASSWORD", "cricintel2024")
    
    # Init session state
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if st.session_state["authenticated"]:
        return True

    # Login screen
    st.markdown("""
    <style>
    .login-container {
        max-width: 420px;
        margin: 8vh auto;
        background: linear-gradient(135deg, #0a0f1e, #0d2137);
        border: 1px solid #00d4ff33;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
    }
    .login-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00d4ff;
        letter-spacing: 0.15em;
        text-shadow: 0 0 20px #00d4ff66;
        margin-bottom: 0.3rem;
    }
    .login-sub {
        color: #7ba7c4;
        font-size: 0.9rem;
        margin-bottom: 1.8rem;
    }
    .login-badge {
        display: inline-block;
        background: #00d4ff22;
        border: 1px solid #00d4ff44;
        color: #00d4ff;
        font-size: 0.72rem;
        padding: 3px 12px;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        letter-spacing: 0.08em;
    }
    </style>
    <div class="login-container">
        <div class="login-title">🏏 CRICINTEL</div>
        <div class="login-sub">AI Cricket Analytics Platform</div>
        <div class="login-badge">PRIVATE BETA</div>
    </div>
    """, unsafe_allow_html=True)

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("####")
        password = st.text_input(
            "Access Code",
            type="password",
            placeholder="Enter your access code",
            key="login_password"
        )
        login_btn = st.button("Access CricIntel →", type="primary", use_container_width=True)
        st.markdown(
            '<div style="text-align:center;margin-top:0.8rem;font-size:0.75rem;color:#4a7a9b;">'
            'Request access: cricintel.net</div>',
            unsafe_allow_html=True
        )

        if login_btn:
            if password == correct_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect access code. Please try again.")

    st.stop()
    return False

# Run login check — stops here if not authenticated
check_login()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 0.5rem;'>
        <div style='font-size:2rem;'>🏏</div>
        <div style='font-size:1.4rem;font-weight:800;color:#00d4ff;letter-spacing:0.12em;'>CRICINTEL</div>
        <div style='font-size:0.75rem;color:#4a7a9b;margin-top:0.3rem;'>AI Cricket Analytics Platform</div>
    </div>
    <hr style='border-color:#1e3a5f;margin:1rem 0;'>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Select Mode",
        ["🔍 Scout Mode","💰 Auction Mode","🎯 Custom Intelligence","🎬 Highlights Generator"],
        index=0
    )

    # ── DATA STATUS IN SIDEBAR ─────────────────────────────────────────
    st.markdown("<hr style='border-color:#1e3a5f;margin:0.8rem 0;'>", unsafe_allow_html=True)
    if st.session_state.get("data_loaded"):
        n_players = len(st.session_state.get("df_master", []))
        st.markdown(f"""
        <div style='background:#0d2137;border:1px solid #00d4ff33;border-radius:8px;padding:0.7rem;margin-bottom:0.5rem;'>
            <div style='color:#00d4ff;font-size:0.78rem;font-weight:600;'>✅ DATA LOADED</div>
            <div style='color:#7ba7c4;font-size:0.72rem;margin-top:0.2rem;'>{n_players} players ready</div>
            <div style='color:#7ba7c4;font-size:0.72rem;'>Switch modes freely ↓</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Upload New Data", use_container_width=True, key="sidebar_reset"):
            for key in ["data_loaded","df_master","players_raw","perf_raw",
                        "contracts_raw","budget_raw","df_processed"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        st.markdown("""
        <div style='background:#1a0d0d;border:1px solid #f8717133;border-radius:8px;padding:0.7rem;margin-bottom:0.5rem;'>
            <div style='color:#f87171;font-size:0.78rem;font-weight:600;'>⬆ No data loaded</div>
            <div style='color:#7ba7c4;font-size:0.72rem;margin-top:0.2rem;'>Upload once, use everywhere</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#1e3a5f;margin:1rem 0;'>
    <div style='font-size:0.75rem;color:#4a7a9b;text-align:center;padding-bottom:1rem;'>
        v4.0 &nbsp;|&nbsp; Built with Streamlit<br>
        <span style='color:#1e3a5f;'>─────────────────</span><br>
        🧠 Upload once · Use everywhere<br>
        📊 Any CSV format supported<br>
        🌍 Any team · Any format
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# UNIFIED DATA UPLOAD SYSTEM
# ═══════════════════════════════════════════════════════
def show_unified_upload():
    """Central upload screen — shown once, data shared across all modes."""
    st.markdown("""
    <div style='max-width:700px;margin:4vh auto;text-align:center;'>
        <div style='font-size:3rem;margin-bottom:1rem;'>🏏</div>
        <div style='font-size:1.8rem;font-weight:800;color:#00d4ff;letter-spacing:0.1em;margin-bottom:0.5rem;'>
            Welcome to CricIntel
        </div>
        <div style='color:#7ba7c4;font-size:1rem;margin-bottom:2rem;'>
            Upload your data once. Scout, analyse, optimise — all in one platform.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📁 Upload Your Data</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        players_f = st.file_uploader(
            "Player data CSV",
            type="csv", key="unified_players",
            help="Any CSV with player names, roles, and basic info"
        )
    with c2:
        perf_f = st.file_uploader(
            "Performance / Metrics CSV",
            type="csv", key="unified_perf",
            help="Any CSV with stats or metrics — cricket or custom"
        )

    # Optional files for Auction Mode
    with st.expander("➕ Add Contracts & Budget (for Auction Mode — optional)"):
        ac1, ac2 = st.columns(2)
        with ac1:
            contracts_f = st.file_uploader("Contracts / Salary CSV", type="csv", key="unified_contracts")
        with ac2:
            budget_f = st.file_uploader("Budget CSV", type="csv", key="unified_budget")

    if not all([players_f, perf_f]):
        st.markdown("""
        <div class="mapper-card" style="margin-top:1.5rem;">
            <div class="mc-title">💡 What can I upload?</div>
            <div class="mc-sub">
                <b>Player CSV:</b> Any spreadsheet with player names and roles.<br>
                <b>Performance CSV:</b> Any spreadsheet with stats or metrics — cricket stats, 
                fitness scores, technical ratings, internal coaching data — anything works.<br><br>
                <b>CricIntel detects your columns automatically.</b> No reformatting needed.
                Once uploaded, all modes — Scout, Auction, Custom Intelligence — use the same data instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return False

    # Process and store in session state
    with st.spinner("🧠 CricIntel is analysing your data..."):
        players = pd.read_csv(players_f)
        perf    = pd.read_csv(perf_f)

        # Store raw files
        st.session_state["players_raw"] = players
        st.session_state["perf_raw"]    = perf

        # Store optional files
        if contracts_f:
            st.session_state["contracts_raw"] = pd.read_csv(contracts_f)
        if budget_f:
            st.session_state["budget_raw"] = pd.read_csv(budget_f)

        # Build master dataframe
        df = build_base_df(players, perf)
        df = compute_phase_scores(df)

        st.session_state["df_master"]    = df
        st.session_state["data_loaded"]  = True

    # Auto redirect to Scout Mode after loading
    st.rerun()

    return True


mode_labels = {
    "🔍 Scout Mode":           ("🔍 Scout Mode",           "Talent identification · Similarity search · Gap-fill recommendations"),
    "💰 Auction Mode":         ("💰 Auction Mode",          "Squad optimisation · Fair salary · Best XI selection"),
    "🎯 Custom Intelligence":  ("🎯 Custom Intelligence",   "Your metrics · Your weights · Your analysis"),
    "🎬 Highlights Generator": ("🎬 Highlights Generator",  "Upload match video · Generate highlight clips"),
}

# ── ROUTING ───────────────────────────────────────────────────────────
if mode == "🎬 Highlights Generator":
    st.markdown(f"""
    <div class="cricintel-banner">
        <div><h1>CRICINTEL</h1><p>Upload match video · Generate highlight clips</p></div>
        <div style='text-align:right;'>
            <div style='font-size:1.1rem;font-weight:700;color:#00d4ff;'>🎬 Highlights Generator</div>
            <div style='font-size:0.78rem;color:#4a7a9b;margin-top:0.3rem;'>AI Cricket Analytics Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    run_highlights_mode()

elif not st.session_state.get("data_loaded"):
    # No data yet — show upload screen only
    st.markdown("""
    <div class="cricintel-banner">
        <div><h1>CRICINTEL</h1><p>Upload your data once · Scout, analyse, optimise — all in one platform</p></div>
        <div style='text-align:right;'>
            <div style='font-size:1.1rem;font-weight:700;color:#00d4ff;'>📁 Upload Data</div>
            <div style='font-size:0.78rem;color:#4a7a9b;margin-top:0.3rem;'>Step 1 of 1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    show_unified_upload()

else:
    # Data loaded — show selected mode with its banner
    banner_title, banner_sub = mode_labels[mode]
    st.markdown(f"""
    <div class="cricintel-banner">
        <div><h1>CRICINTEL</h1><p>{banner_sub}</p></div>
        <div style='text-align:right;'>
            <div style='font-size:1.1rem;font-weight:700;color:#00d4ff;'>{banner_title}</div>
            <div style='font-size:0.78rem;color:#4a7a9b;margin-top:0.3rem;'>AI Cricket Analytics Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if mode == "🔍 Scout Mode":
        run_scout_mode()
    elif mode == "💰 Auction Mode":
        run_auction_mode()
    elif mode == "🎯 Custom Intelligence":
        run_custom_intelligence()
