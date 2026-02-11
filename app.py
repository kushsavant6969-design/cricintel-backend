import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

import pulp as pl

# NEW imports for Highlights Generator
import os
import tempfile
import subprocess


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Salary Optimizer", layout="wide")
st.title("CRICINTEL")
st.caption("Two modes: Scout Mode (2 CSV) and Auction Mode (4 CSV).")

# =========================================================
# MODE SELECTOR  (UPDATED: added 3rd option)
# =========================================================
mode = st.sidebar.radio(
    "Select Mode",
    ["Scout Mode (Light)", "Auction Mode (Full)", "Highlights Generator"],
    index=0
)

# =========================================================
# HELPERS
# =========================================================
def nz(s: pd.Series, default=0.0):
    if s is None:
        return pd.Series(dtype=float)
    return s.fillna(default)

def norm01(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    mn, mx = s.min(), s.max()
    if mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)

def clamp01(x):
    return np.clip(x, 0, 1)

def to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8")

def stable_noise_from_id(id_series: pd.Series, mod=97) -> np.ndarray:
    s = pd.to_numeric(id_series, errors="coerce").fillna(0).astype(int)
    return (s % mod) / float(mod)

def topn(df, col, n=10, asc=False):
    return df.sort_values(col, ascending=asc).head(n)

def safe_numeric(df, cols, default=0.0):
    for c in cols:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    return df


# =========================================================
# OPTIMISERS
# =========================================================
def optimize_squad_soft(df: pd.DataFrame,
                        budget_limit: float,
                        max_players: int,
                        min_role: dict,
                        price_col: str,
                        extra_min_flags: dict | None = None,
                        extra_max_flags: dict | None = None,
                        lock_players: set | None = None,
                        max_single_price: float | None = None,
                        penalty_weight: float = 2.5,
                        price_concentration_penalty: float = 0.0):
    """
    Soft mins: extra_min_flags become soft with slack variables.
    Role mins remain hard.
    Adds optional: max_single_price constraint + price concentration penalty.
    """
    d = df.copy()
    players_list = d["player"].tolist()
    if not players_list:
        return pd.DataFrame(), {"status": "empty"}

    x = pl.LpVariable.dicts("pick", players_list, lowBound=0, upBound=1, cat="Binary")
    prob = pl.LpProblem("SquadSelectionSoft", pl.LpMaximize)

    slack = {}
    if extra_min_flags:
        for col, mn in extra_min_flags.items():
            if col in d.columns and int(mn) > 0:
                slack[col] = pl.LpVariable(f"slack_{col}", lowBound=0, cat="Continuous")

    norm_price = norm01(d[price_col]).values

    prob += (
        pl.lpSum(d.loc[d.player == p, "objective_score"].values[0] * x[p] for p in players_list)
        - float(penalty_weight) * pl.lpSum(slack[c] for c in slack)
        - float(price_concentration_penalty) * pl.lpSum(norm_price[i] * x[p] for i, p in enumerate(players_list))
    )

    prob += pl.lpSum(d.loc[d.player == p, price_col].values[0] * x[p] for p in players_list) <= float(budget_limit)
    prob += pl.lpSum(x[p] for p in players_list) <= int(max_players)

    def role_count(code: str):
        return pl.lpSum(x[p] for p in players_list if d.loc[d.player == p, "role"].values[0] == code)

    for r, mn in min_role.items():
        prob += role_count(r) >= int(mn)

    if extra_min_flags:
        for col, mn in extra_min_flags.items():
            if col in d.columns and int(mn) > 0:
                cnt = pl.lpSum(x[p] for p in players_list if int(d.loc[d.player == p, col].values[0]) == 1)
                prob += cnt + slack[col] >= int(mn)

    if extra_max_flags:
        for col, mx in extra_max_flags.items():
            if col in d.columns and mx is not None:
                prob += pl.lpSum(x[p] for p in players_list if int(d.loc[d.player == p, col].values[0]) == 1) <= int(mx)

    if max_single_price is not None:
        for p in players_list:
            prob += d.loc[d.player == p, price_col].values[0] * x[p] <= float(max_single_price)

    if lock_players:
        for lp in lock_players:
            if lp in players_list:
                prob += x[lp] == 1

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    picked = [p for p in players_list if x[p].value() == 1]
    squad = d[d.player.isin(picked)].copy()

    violations = {}
    for col in slack:
        try:
            violations[col] = float(slack[col].value())
        except Exception:
            violations[col] = None

    metrics = {
        "status": "ok" if len(squad) else "no_solution",
        "count": int(len(squad)),
        "spend": float(squad[price_col].sum()) if len(squad) else 0.0,
        "objective": float(squad["objective_score"].sum()) if len(squad) else 0.0,
        "violations": violations
    }
    for r in ["BAT", "BOWL", "AR", "WK"]:
        metrics[r] = int((squad["role"] == r).sum()) if len(squad) else 0

    return squad, metrics


def pick_best_xi(squad: pd.DataFrame,
                 xi_size: int,
                 xi_min_role: dict,
                 max_overseas_xi: int | None,
                 enforce_left_in_top4: bool):
    if len(squad) == 0:
        return pd.DataFrame(), {"status": "empty"}

    d = squad.copy()
    players_list = d["player"].tolist()

    x = pl.LpVariable.dicts("xi", players_list, lowBound=0, upBound=1, cat="Binary")
    prob = pl.LpProblem("BestXI", pl.LpMaximize)

    prob += pl.lpSum(d.loc[d.player == p, "xi_score"].values[0] * x[p] for p in players_list)
    prob += pl.lpSum(x[p] for p in players_list) == int(xi_size)

    def role_count(code: str):
        return pl.lpSum(x[p] for p in players_list if d.loc[d.player == p, "role"].values[0] == code)

    for r, mn in xi_min_role.items():
        prob += role_count(r) >= int(mn)

    if max_overseas_xi is not None and "is_overseas" in d.columns:
        prob += pl.lpSum(x[p] for p in players_list if int(d.loc[d.player == p, "is_overseas"].values[0]) == 1) <= int(max_overseas_xi)

    if enforce_left_in_top4 and "bat_hand" in d.columns and "is_top4_candidate" in d.columns:
        prob += pl.lpSum(
            x[p] for p in players_list
            if (int(d.loc[d.player == p, "is_top4_candidate"].values[0]) == 1 and str(d.loc[d.player == p, "bat_hand"].values[0]) == "L")
        ) >= 1

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    picked = [p for p in players_list if x[p].value() == 1]
    xi = d[d.player.isin(picked)].copy()

    metrics = {"status": "ok" if len(xi) else "no_solution", "count": int(len(xi)),
               "xi_score": float(xi["xi_score"].sum()) if len(xi) else 0.0}
    return xi, metrics


# =========================================================
# CORE DATA PIPELINE (used by BOTH modes)
# =========================================================
def build_base_df(players: pd.DataFrame, perf: pd.DataFrame, contracts: pd.DataFrame | None = None):
    df = players.merge(perf, on="player_id", how="left")
    if contracts is not None:
        df = df.merge(contracts, on="player_id", how="left")

    # required base
    req_players = {"player", "player_id", "role", "age"}
    req_perf = {"player_id", "matches", "runs", "strike_rate", "wickets", "economy", "dot_ball_pct", "boundary_pct"}

    if not req_players.issubset(players.columns):
        missing = sorted(list(req_players - set(players.columns)))
        st.error(f"players.csv missing: {missing}")
        st.stop()

    if not req_perf.issubset(perf.columns):
        missing = sorted(list(req_perf - set(perf.columns)))
        st.error(f"performance.csv missing: {missing}")
        st.stop()

    # numeric safety
    df = safe_numeric(df, ["matches","runs","strike_rate","wickets","economy","dot_ball_pct","boundary_pct","age"], 0.0)

    # optional columns
    for col in ["pp_sr","middle_sr","death_sr","pp_runs","death_runs",
                "pp_eco","middle_eco","death_eco","pp_wkts","death_wkts",
                "injury_risk","availability_risk"]:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["is_pp_bowler","is_death_bowler","is_pp_batter","is_death_hitter",
                "is_spinner","is_pacer","is_overseas"]:
        if col not in df.columns:
            df[col] = 0

    if "bat_hand" not in df.columns:
        df["bat_hand"] = "R"

    # NEW (inferred if missing)
    if "bowling_arm" not in df.columns:
        df["bowling_arm"] = np.nan   # "L" / "R"
    if "spin_type" not in df.columns:
        df["spin_type"] = np.nan     # "OFF" / "LEG" / "NONE"

    # deterministic phase proxies if missing
    noise = stable_noise_from_id(df["player_id"])

    if df["pp_sr"].isna().all():
        df["pp_sr"] = df["strike_rate"] + (-2 + 6*(noise - 0.5))
        df["middle_sr"] = df["strike_rate"] + (-6 + 8*(noise - 0.5))
        df["death_sr"] = df["strike_rate"] + (8 + 10*(noise - 0.5))
        df["pp_runs"] = (df["runs"] * (0.33 + 0.06*(noise - 0.5))).astype(float)
        df["death_runs"] = (df["runs"] * (0.23 + 0.06*(noise - 0.5))).astype(float)

    if df["pp_eco"].isna().all():
        df["pp_eco"] = df["economy"] + (-0.25 + 0.6*(noise - 0.5))
        df["middle_eco"] = df["economy"] + (0.00 + 0.5*(noise - 0.5))
        df["death_eco"] = df["economy"] + (0.55 + 0.8*(noise - 0.5))
        df["pp_wkts"] = (df["wickets"] * (0.33 + 0.05*(noise - 0.5))).astype(float)
        df["death_wkts"] = (df["wickets"] * (0.23 + 0.05*(noise - 0.5))).astype(float)

    # infer roles
    if "batting_role" not in df.columns:
        df["batting_role"] = "ANCHOR"
    if "bowling_role" not in df.columns:
        df["bowling_role"] = "MIDDLE"

    is_bat_candidate = df["role"].isin(["BAT", "AR", "WK"])
    opener_mask = is_bat_candidate & (df["pp_sr"] >= df["pp_sr"].quantile(0.70))
    finisher_mask = is_bat_candidate & (df["death_sr"] >= df["death_sr"].quantile(0.75))
    df.loc[is_bat_candidate, "batting_role"] = "ANCHOR"
    df.loc[opener_mask, "batting_role"] = "OPENER"
    df.loc[finisher_mask, "batting_role"] = "FINISHER"

    is_bowl_candidate = df["role"].isin(["BOWL", "AR"])
    death_mask = is_bowl_candidate & ((df["death_wkts"] >= df["death_wkts"].quantile(0.70)) | (df["is_death_bowler"].astype(int) == 1))
    pp_mask = is_bowl_candidate & ((df["pp_wkts"] >= df["pp_wkts"].quantile(0.65)) | (df["is_pp_bowler"].astype(int) == 1))
    df.loc[is_bowl_candidate, "bowling_role"] = "MIDDLE"
    df.loc[pp_mask, "bowling_role"] = "PP"
    df.loc[death_mask, "bowling_role"] = "DEATH"

    df["is_opener"] = (df["batting_role"] == "OPENER").astype(int)
    df["is_finisher"] = (df["batting_role"] == "FINISHER").astype(int)
    df["is_death_bowler2"] = (df["bowling_role"] == "DEATH").astype(int)
    df["is_pp_bowler2"] = (df["bowling_role"] == "PP").astype(int)

    df["is_top4_candidate"] = (
        (df["role"].isin(["BAT", "WK", "AR"])) &
        ((df["is_opener"] == 1) | (df["runs"] >= df["runs"].quantile(0.60)))
    ).astype(int)

    # NEW: infer bowling arm + spin type if missing
    if df["bowling_arm"].isna().all():
        df["bowling_arm"] = np.where(noise < 0.28, "L", "R")
    df["bowling_arm"] = df["bowling_arm"].astype(str).str.upper()
    df.loc[~df["bowling_arm"].isin(["L", "R"]), "bowling_arm"] = "R"

    if df["spin_type"].isna().all():
        df["spin_type"] = "NONE"
        spin_mask = df["is_spinner"].astype(int) == 1
        df.loc[spin_mask, "spin_type"] = np.where(noise[spin_mask] < 0.55, "OFF", "LEG")
    df["spin_type"] = df["spin_type"].astype(str).str.upper()

    # helper flags for coach
    df["is_left_arm_spinner"] = ((df["is_spinner"].astype(int)==1) & (df["bowling_arm"]=="L")).astype(int)
    df["is_right_arm_spinner"] = ((df["is_spinner"].astype(int)==1) & (df["bowling_arm"]=="R")).astype(int)
    df["is_off_spinner"] = ((df["is_spinner"].astype(int)==1) & (df["spin_type"]=="OFF")).astype(int)
    df["is_leg_spinner"] = ((df["is_spinner"].astype(int)==1) & (df["spin_type"]=="LEG")).astype(int)
    df["is_left_arm_pacer"] = ((df["is_pacer"].astype(int)==1) & (df["bowling_arm"]=="L")).astype(int)
    df["is_right_arm_pacer"] = ((df["is_pacer"].astype(int)==1) & (df["bowling_arm"]=="R")).astype(int)

    # risk model (safe for both modes)
    if df["injury_risk"].isna().all():
        workload = norm01(df["matches"])
        inj_proxy = 0.12 + 0.02*(df["age"] - df["age"].min()) + 0.20*workload + 0.10*df["is_pacer"].astype(float)
        df["injury_risk"] = clamp01(inj_proxy)

    if df["availability_risk"].isna().all():
        workload = norm01(df["matches"])
        av_proxy = 0.08 + 0.15*df["is_overseas"].astype(float) + 0.10*workload
        df["availability_risk"] = clamp01(av_proxy)

    df["total_risk"] = clamp01(0.65*df["injury_risk"].astype(float) + 0.35*df["availability_risk"].astype(float))
    return df


# =========================================================
# SCOUT MODE (2 CSV) + GAPFILL
# =========================================================
def run_scout_mode():
    st.subheader("Scout Mode")
    st.caption("Upload players.csv + performance.csv. No contracts. Focus: profiling + similarity + gap-fill recommendations.")

    c1, c2 = st.columns(2)
    with c1:
        players_f = st.file_uploader("players.csv", type="csv", key="sc_pl")
    with c2:
        performance_f = st.file_uploader("performance.csv", type="csv", key="sc_pf")

    if not all([players_f, performance_f]):
        st.info("Upload players.csv and performance.csv to proceed.")
        st.stop()

    players = pd.read_csv(players_f)
    perf = pd.read_csv(performance_f)

    df = build_base_df(players, perf, contracts=None)

    st.subheader("Quick Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Players", len(df))
    k2.metric("Batters", int((df["role"] == "BAT").sum()))
    k3.metric("Bowlers", int((df["role"] == "BOWL").sum()))
    k4.metric("All-rounders", int((df["role"] == "AR").sum()))

    # Phase scores (also used for gap-fill)
    w_pp, w_mid, w_death = 1.0, 1.0, 1.2

    pp_bat = 0.6*norm01(df["pp_sr"]) + 0.4*norm01(df["pp_runs"])
    mid_bat = norm01(df["middle_sr"])
    death_bat = 0.6*norm01(df["death_sr"]) + 0.4*norm01(df["death_runs"])

    pp_bowl = 0.6*norm01(df["pp_wkts"]) + 0.4*(1 - norm01(df["pp_eco"]))
    mid_bowl = 0.7*(1 - norm01(df["middle_eco"])) + 0.3*norm01(df["dot_ball_pct"])
    death_bowl = 0.6*norm01(df["death_wkts"]) + 0.4*(1 - norm01(df["death_eco"]))

    df["pp_bat_score"] = pp_bat
    df["mid_bat_score"] = mid_bat
    df["death_bat_score"] = death_bat
    df["pp_bowl_score"] = pp_bowl
    df["mid_bowl_score"] = mid_bowl
    df["death_bowl_score"] = death_bowl

    # role-aware impact score
    impact = []
    for r, b, w in zip(df["role"].fillna("BAT"), (w_pp*pp_bat + w_mid*mid_bat + w_death*death_bat),
                       (w_pp*pp_bowl + w_mid*mid_bowl + w_death*death_bowl)):
        if r == "BAT":
            impact.append(b)
        elif r == "BOWL":
            impact.append(w)
        elif r == "AR":
            impact.append(0.55*b + 0.45*w)
        elif r == "WK":
            impact.append(0.95*b + 0.05)
        else:
            impact.append(b)
    df["match_impact_score"] = norm01(pd.Series(impact))

    st.subheader("Top Profiles")
    st.dataframe(
        df[["player","role","age","bat_hand","bowling_arm","spin_type","batting_role","bowling_role",
            "matches","runs","strike_rate","wickets","economy","dot_ball_pct","boundary_pct",
            "match_impact_score","total_risk"]]
        .sort_values("match_impact_score", ascending=False)
        .head(25),
        use_container_width=True
    )

    # Similarity search
    st.subheader("Similarity Search")
    st.caption("Pick a player and find closest matches using phase profile + core indicators.")
    target = st.selectbox("Select a player", df["player"].tolist(), index=0)

    def get_similar_players(df_all: pd.DataFrame, target_name: str, top_k: int = 10):
        trow = df_all[df_all["player"] == target_name].head(1)
        if len(trow) == 0:
            return pd.DataFrame()

        feat_cols = [
            "pp_bat_score","mid_bat_score","death_bat_score",
            "pp_bowl_score","mid_bowl_score","death_bowl_score",
            "strike_rate","economy","dot_ball_pct","boundary_pct","match_impact_score"
        ]
        feat_cols = [c for c in feat_cols if c in df_all.columns]

        X = df_all[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        t = trow[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

        sims = cosine_similarity(X, t).reshape(-1)
        out = df_all.copy()
        out["similarity"] = sims
        out = out[out["player"] != target_name]
        out = out.sort_values("similarity", ascending=False).head(top_k)
        return out

    sim = get_similar_players(df, target, top_k=12)
    if len(sim):
        st.dataframe(
            sim[["player","role","batting_role","bowling_role","similarity","match_impact_score","total_risk"]]
            .sort_values("similarity", ascending=False),
            use_container_width=True
        )

    # =========================================================
    # STEP 1 — UNIVERSAL GAP-FILL (1/2/3+ supported)
    # =========================================================
    st.subheader("Step 1: Universal Gap-fill Recommender (Any Role)")
    st.caption("If you select 1 player → player-style match. 2 players → small-unit. 3+ → full unit profile.")

    gap_options = [
        "Pacer (any)", "Left-arm Pacer", "Right-arm Pacer",
        "Spinner (any)", "Left-arm Spinner", "Right-arm Spinner",
        "Off-spinner", "Leg-spinner",
        "Opener", "Top-4 (anchor)", "Finisher"
    ]
    gap_type = st.selectbox("What gap are you trying to fill?", gap_options, index=0)
    allow_small = st.checkbox("Allow small unit (≥2) — less stable", value=True)

    # pool
    def gap_pool(df_all: pd.DataFrame, gap: str) -> pd.DataFrame:
        d = df_all.copy()
        if gap == "Pacer (any)":
            return d[d["is_pacer"].astype(int) == 1]
        if gap == "Left-arm Pacer":
            return d[d["is_left_arm_pacer"].astype(int) == 1]
        if gap == "Right-arm Pacer":
            return d[d["is_right_arm_pacer"].astype(int) == 1]
        if gap == "Spinner (any)":
            return d[d["is_spinner"].astype(int) == 1]
        if gap == "Left-arm Spinner":
            return d[d["is_left_arm_spinner"].astype(int) == 1]
        if gap == "Right-arm Spinner":
            return d[d["is_right_arm_spinner"].astype(int) == 1]
        if gap == "Off-spinner":
            return d[d["is_off_spinner"].astype(int) == 1]
        if gap == "Leg-spinner":
            return d[d["is_leg_spinner"].astype(int) == 1]
        if gap == "Opener":
            return d[d["is_opener"].astype(int) == 1]
        if gap == "Top-4 (anchor)":
            return d[d["is_top4_candidate"].astype(int) == 1]
        if gap == "Finisher":
            return d[d["is_finisher"].astype(int) == 1]
        return d

    pool_df = gap_pool(df, gap_type)
    if len(pool_df) == 0:
        st.warning("No players found for this gap type (check your flags/columns).")
        st.stop()

    current_players = st.multiselect(
        f"Select your CURRENT players for: {gap_type}",
        options=pool_df["player"].tolist(),
        default=[]
    )

    nsel = len(current_players)
    if nsel == 0:
        st.info("Select at least 1 player to get recommendations.")
        st.stop()

    if nsel == 1:
        st.info("Mode: Player-style match (1 selected)")
    elif nsel == 2:
        if not allow_small:
            st.warning("Select 3+ players for full unit match, or enable 'Allow small unit (≥2)'.")
            st.stop()
        st.info("Mode: Small unit match (2 selected) — less stable")
    else:
        st.success("Mode: Unit match (3+ selected) — best stability")

    def features_for_gap(gap: str) -> list[str]:
        if gap in ["Pacer (any)", "Left-arm Pacer", "Right-arm Pacer",
                   "Spinner (any)", "Left-arm Spinner", "Right-arm Spinner",
                   "Off-spinner", "Leg-spinner"]:
            feats = [
                "pp_bowl_score","mid_bowl_score","death_bowl_score",
                "pp_eco","middle_eco","death_eco",
                "pp_wkts","death_wkts",
                "economy","dot_ball_pct","match_impact_score","total_risk"
            ]
            return [c for c in feats if c in df.columns]

        if gap in ["Opener", "Top-4 (anchor)"]:
            feats = [
                "pp_bat_score","mid_bat_score",
                "pp_sr","middle_sr","strike_rate",
                "pp_runs","runs","boundary_pct",
                "match_impact_score","total_risk"
            ]
            return [c for c in feats if c in df.columns]

        if gap in ["Finisher"]:
            feats = [
                "death_bat_score","death_sr","death_runs","strike_rate",
                "boundary_pct","match_impact_score","total_risk"
            ]
            return [c for c in feats if c in df.columns]

        return [c for c in ["match_impact_score","total_risk","strike_rate","economy"] if c in df.columns]

    feat_cols = features_for_gap(gap_type)

    base = df[df["player"].isin(current_players)].copy()
    cand = pool_df[~pool_df["player"].isin(current_players)].copy()
    if len(cand) == 0:
        st.warning("No candidates available outside selected group.")
        st.stop()

    base_feats = base[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cand_feats = cand[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    centroid = base_feats.mean(axis=0).values.reshape(1, -1)
    sims = cosine_similarity(cand_feats.values, centroid).reshape(-1)
    cand["unit_fit_score"] = sims

    cand["combined_reco"] = (
        0.70*cand["unit_fit_score"]
        + 0.30*norm01(cand["match_impact_score"])
        - 0.20*norm01(cand["total_risk"])
    )

    rec = cand.sort_values("combined_reco", ascending=False).head(10)

    # Unit profile table
    unit_profile = base_feats.mean(axis=0).reset_index()
    unit_profile.columns = ["metric", "unit_avg"]

    st.subheader("Unit profile (from selected players)")
    st.dataframe(unit_profile, use_container_width=True)

    st.subheader("Top 10 recommendations")
    show_cols = ["player","role","age","bat_hand","bowling_arm","spin_type","batting_role","bowling_role",
                "unit_fit_score","combined_reco","match_impact_score","total_risk"]
    show_cols = [c for c in show_cols if c in rec.columns]
    st.dataframe(rec[show_cols], use_container_width=True)

    # Explainability
    st.subheader("Explainability (top 5)")
    unit_avg = unit_profile.set_index("metric")["unit_avg"].to_dict()

    def explain_row(row: pd.Series):
        diffs = {}
        for c in feat_cols:
            try:
                diffs[c] = float(row[c]) - float(unit_avg.get(c, 0.0))
            except Exception:
                continue
        ranked = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[:3]
        worst = ranked[-1:] if ranked else []
        return best, worst

    for idx, r in rec.head(5).iterrows():
        best, worst = explain_row(r)
        st.markdown(f"**{r['player']}** — combined_reco: `{r.get('combined_reco',0):.3f}` | unit_fit: `{r.get('unit_fit_score',0):.3f}`")
        for k, v in best:
            st.write(f"• **{k}** improves vs selected ({v:+.2f})")
        for k, v in worst:
            st.write(f"• Trade-off: **{k}** below selected ({v:+.2f})")
        st.write("---")

    # Downloads
    st.subheader("Downloads")
    st.download_button("Download Scout Table (CSV)", to_csv_bytes(df), "scout_mode_table.csv", "text/csv")


# =========================================================
# AUCTION MODE (4 CSV) - full build preserved
# =========================================================
def run_auction_mode():
    st.subheader("Auction Mode")
    st.caption("Upload players.csv + performance.csv + contracts.csv + budget.csv (Full features).")

    c1, c2 = st.columns(2)
    with c1:
        players_f = st.file_uploader("players.csv", type="csv", key="au_pl")
        performance_f = st.file_uploader("performance.csv", type="csv", key="au_pf")
    with c2:
        contracts_f = st.file_uploader("contracts.csv", type="csv", key="au_ct")
        budget_f = st.file_uploader("budget.csv", type="csv", key="au_bd")

    if not all([players_f, performance_f, contracts_f, budget_f]):
        st.info("Upload all four CSVs to proceed.")
        st.stop()

    players = pd.read_csv(players_f)
    perf = pd.read_csv(performance_f)
    contracts = pd.read_csv(contracts_f)
    budget_row = pd.read_csv(budget_f).iloc[0]

    # contracts required col
    req_contract = {"player_id", "current_salary_lakh"}
    if not req_contract.issubset(contracts.columns):
        st.error(f"contracts.csv missing: {sorted(list(req_contract - set(contracts.columns)))}")
        st.stop()

    contracts["current_salary_lakh"] = pd.to_numeric(contracts["current_salary_lakh"], errors="coerce").fillna(0.0)

    df = build_base_df(players, perf, contracts=contracts)
    df["current_salary_lakh"] = pd.to_numeric(df["current_salary_lakh"], errors="coerce").fillna(0.0)

    # Context settings
    st.subheader("Context Settings")
    cc1, cc2, cc3, cc4 = st.columns(4)
    pitch_type = cc1.selectbox("Home pitch type", ["Flat/True", "Spin-friendly", "Pace/Bounce"], index=0)
    season_goal = cc2.selectbox("Season strategy", ["Balanced", "Batting dominance", "Bowling dominance"], index=0)
    risk_pref = cc3.selectbox("Risk preference", ["Balanced", "Risk-averse", "High-upside"], index=0)
    auction_mode = cc4.checkbox("Auction Mode", value=True)

    # Opponent profile
    st.subheader("Opponent profile")
    opp_profiles = {
        "Balanced opponent": {"spin": 0.5, "pace": 0.5, "death_bowl": 0.5, "pp_aggr": 0.5},
        "Spin choke team (middle overs)": {"spin": 0.85, "pace": 0.35, "death_bowl": 0.55, "pp_aggr": 0.45},
        "Pace/bounce attack team": {"spin": 0.35, "pace": 0.85, "death_bowl": 0.65, "pp_aggr": 0.55},
        "Death-overs specialists": {"spin": 0.45, "pace": 0.55, "death_bowl": 0.90, "pp_aggr": 0.45},
        "Powerplay smashers": {"spin": 0.45, "pace": 0.55, "death_bowl": 0.55, "pp_aggr": 0.90},
    }
    oc1, oc2 = st.columns([2, 1])
    opp_profile = oc1.selectbox("Opponent type", list(opp_profiles.keys()), index=0)
    manual_override = oc2.checkbox("Manual override sliders", value=False)

    base = opp_profiles[opp_profile]
    o1, o2, o3, o4 = st.columns(4)
    opp_spin_heavy = o1.slider("Opponent spin-heavy", 0.0, 1.0, float(base["spin"]), 0.05, disabled=not manual_override)
    opp_pace_bounce = o2.slider("Opponent pace/bounce threat", 0.0, 1.0, float(base["pace"]), 0.05, disabled=not manual_override)
    opp_death_bowling_strong = o3.slider("Opponent death bowling strong", 0.0, 1.0, float(base["death_bowl"]), 0.05, disabled=not manual_override)
    opp_powerplay_aggressive = o4.slider("Opponent powerplay aggressive", 0.0, 1.0, float(base["pp_aggr"]), 0.05, disabled=not manual_override)

    # Auction settings
    st.subheader("Auction settings")
    a1, a2, a3 = st.columns(3)
    inflation = a1.slider("Auction inflation", 1.0, 2.5, 1.35, 0.05, disabled=not auction_mode)
    reserve_floor = a2.number_input("Reserve floor (₹ lakh)", value=120.0, step=10.0, disabled=not auction_mode)
    budget_lakh = a3.number_input("Budget (₹ lakh)", value=float(budget_row.get("budget_lakh", 10000)), step=100.0)

    # Squad constraints
    st.subheader("Squad constraints")
    c5, c6, c7, c8 = st.columns(4)
    max_players = c5.number_input("Max squad size", value=int(budget_row.get("max_players", 25)), step=1)
    min_bat = c6.number_input("Min BAT", value=int(budget_row.get("min_bat", 8)), step=1)
    min_bowl = c7.number_input("Min BOWL", value=int(budget_row.get("min_bowl", 8)), step=1)
    min_ar = c8.number_input("Min AR", value=int(budget_row.get("min_ar", 4)), step=1)

    c9, c10 = st.columns(2)
    min_wk = c9.number_input("Min WK", value=int(budget_row.get("min_wk", 2)), step=1)
    max_overseas_squad = c10.number_input("Max Overseas (squad)", value=8, step=1)

    min_role = {"BAT": min_bat, "BOWL": min_bowl, "AR": min_ar, "WK": min_wk}

    # Balance constraints
    st.subheader("Balance constraints")
    b1, b2, b3, b4 = st.columns(4)
    min_spinners = b1.number_input("Min Spinners (squad)", value=3, step=1)
    min_pacers = b2.number_input("Min Pacers (squad)", value=3, step=1)
    min_death_bowl = b3.number_input("Min Death Bowlers", value=2, step=1)
    min_death_hit = b4.number_input("Min Death Hitters", value=2, step=1)

    b5, b6, b7 = st.columns(3)
    min_pp_bowl = b5.number_input("Min Powerplay Bowlers", value=2, step=1)
    min_openers = b6.number_input("Min Openers (squad)", value=2, step=1)
    min_finishers = b7.number_input("Min Finishers (squad)", value=2, step=1)

    enforce_left_in_top4 = st.checkbox("Enforce left-right balance (≥1 lefty top-4)", value=True)

    # Always feasible
    st.subheader("Always-feasible mode")
    soft_mode = st.checkbox("Use Soft Constraints (always returns best squad)", value=True)
    penalty_weight = st.slider("Soft constraint penalty (higher = stricter)", 0.5, 10.0, 2.5, 0.1, disabled=not soft_mode)

    # Retentions
    st.subheader("Retentions / RTM (simulation)")
    ret_col1, ret_col2 = st.columns([2, 1])
    retained_players = ret_col1.multiselect("Select retained/locked players", df["player"].tolist(), default=[])
    default_ret_cost = ret_col2.number_input("Default retained cost (₹ lakh each)", value=600.0, step=50.0)

    retained_costs = {}
    if retained_players:
        st.write("Set retained cost per player (optional):")
        for p in retained_players[:25]:
            retained_costs[p] = st.number_input(f"{p} retained cost", value=float(default_ret_cost), step=50.0)

    locked_set = set(retained_players)
    retained_total = sum(retained_costs.get(p, default_ret_cost) for p in retained_players)
    budget_after_ret = float(budget_lakh - retained_total)

    st.metric("Budget after retentions", f"{budget_after_ret:.1f} lakh")
    if budget_after_ret <= 0:
        st.error("Retentions exceed budget. Reduce retained costs or raise budget.")
        st.stop()

    # Auction price used
    df["auction_price_lakh"] = df["current_salary_lakh"].copy()
    if auction_mode:
        df["auction_price_lakh"] = (df["current_salary_lakh"] * float(inflation)).clip(lower=float(reserve_floor))
    price_col = "auction_price_lakh" if auction_mode else "current_salary_lakh"
    df["price_used_lakh"] = df[price_col].copy()

    if retained_players:
        df["price_used_lakh"] = df["price_used_lakh"].astype(float)
        for p in retained_players:
            df.loc[df["player"] == p, "price_used_lakh"] = float(retained_costs.get(p, default_ret_cost))
    price_col = "price_used_lakh"

    # Risk preference adjustment
    if risk_pref == "Risk-averse":
        risk_penalty_strength = 1.2
    elif risk_pref == "High-upside":
        risk_penalty_strength = 0.4
    else:
        risk_penalty_strength = 0.8
    df["risk_adjustment"] = (1 - risk_penalty_strength * df["total_risk"]).clip(lower=0.25, upper=1.05)

    # Pitch fit
    fit = np.ones(len(df), dtype=float)
    if pitch_type == "Spin-friendly":
        fit += 0.12*df["is_spinner"].astype(float)
        fit += 0.06*(df["bowling_role"] == "MIDDLE").astype(float)
        fit += 0.05*(df["batting_role"] == "ANCHOR").astype(float)
        fit -= 0.04*df["is_pacer"].astype(float)
    elif pitch_type == "Pace/Bounce":
        fit += 0.12*df["is_pacer"].astype(float)
        fit += 0.06*(df["bowling_role"].isin(["PP", "DEATH"])).astype(float)
        fit += 0.05*norm01(df["pp_sr"])
        fit -= 0.04*df["is_spinner"].astype(float)
    else:
        fit += 0.10*norm01(df["strike_rate"])
        fit += 0.08*(df["batting_role"] == "FINISHER").astype(float)
    df["pitch_fit"] = np.clip(fit, 0.80, 1.30)

    # Phase importance
    st.subheader("Phase importance")
    p1, p2, p3 = st.columns(3)
    w_pp = p1.slider("Powerplay weight", 0.0, 2.0, 1.0, 0.1)
    w_mid = p2.slider("Middle overs weight", 0.0, 2.0, 1.0, 0.1)
    w_death = p3.slider("Death overs weight", 0.0, 2.0, 1.2, 0.1)

    # Phase scores
    pp_bat = 0.6*norm01(df["pp_sr"]) + 0.4*norm01(df["pp_runs"])
    mid_bat = norm01(df["middle_sr"])
    death_bat = 0.6*norm01(df["death_sr"]) + 0.4*norm01(df["death_runs"])

    pp_bowl = 0.6*norm01(df["pp_wkts"]) + 0.4*(1 - norm01(df["pp_eco"]))
    mid_bowl = 0.7*(1 - norm01(df["middle_eco"])) + 0.3*norm01(df["dot_ball_pct"])
    death_bowl = 0.6*norm01(df["death_wkts"]) + 0.4*(1 - norm01(df["death_eco"]))

    df["pp_bat_score"] = pp_bat
    df["mid_bat_score"] = mid_bat
    df["death_bat_score"] = death_bat
    df["pp_bowl_score"] = pp_bowl
    df["mid_bowl_score"] = mid_bowl
    df["death_bowl_score"] = death_bowl

    impact = []
    for r, b, w in zip(df["role"].fillna("BAT"), (w_pp*pp_bat + w_mid*mid_bat + w_death*death_bat),
                       (w_pp*pp_bowl + w_mid*mid_bowl + w_death*death_bowl)):
        if r == "BAT":
            impact.append(b)
        elif r == "BOWL":
            impact.append(w)
        elif r == "AR":
            impact.append(0.55*b + 0.45*w)
        elif r == "WK":
            impact.append(0.95*b + 0.05)
        else:
            impact.append(b)
    df["match_impact_score"] = norm01(pd.Series(impact))

    # Opponent fit
    lefty = (df["bat_hand"].astype(str) == "L").astype(float)
    anchor = (df["batting_role"] == "ANCHOR").astype(float)
    opener = (df["batting_role"] == "OPENER").astype(float)
    pp_bowler_flag = (df["bowling_role"] == "PP").astype(float)
    pacer = df["is_pacer"].astype(float)
    spinner = df["is_spinner"].astype(float)

    opp_mult = np.ones(len(df), dtype=float)
    opp_mult += 0.10 * opp_spin_heavy * (0.9*lefty + 0.6*anchor + 0.6*spinner + 0.2*df["mid_bat_score"])
    opp_mult += 0.10 * opp_pace_bounce * (0.9*pacer + 0.6*opener + 0.5*pp_bowler_flag + 0.2*df["pp_bat_score"])
    opp_mult += 0.08 * opp_death_bowling_strong * (0.9*anchor + 0.4*df["mid_bat_score"])
    opp_mult += 0.08 * opp_powerplay_aggressive * (0.9*pp_bowler_flag + 0.5*(1 - norm01(df["pp_eco"])))
    df["opponent_fit"] = np.clip(opp_mult, 0.85, 1.22)

    # Fair salary regression
    st.subheader("Fair Salary (Regression baseline)")
    feature_pool = ["role", "age", "runs", "wickets", "matches", "strike_rate", "economy", "dot_ball_pct", "boundary_pct", "match_impact_score"]
    X = df[[c for c in feature_pool if c in df.columns]].copy()
    y = df["current_salary_lakh"].copy()

    ct = ColumnTransformer(
        [("role_ohe", OneHotEncoder(handle_unknown="ignore"), ["role"] if "role" in X.columns else [])],
        remainder="passthrough"
    )
    model = Pipeline([("prep", ct), ("reg", Ridge(alpha=1.0))])
    model.fit(X, y)
    df["fair_salary_lakh"] = np.clip(model.predict(X), 0, None)

    # Flex score
    df["phase_versatility"] = (
        0.5*((df["is_pp_batter"].astype(int) == 1) | (df["is_pp_bowler"].astype(int) == 1)).astype(float) +
        0.5*((df["is_death_hitter"].astype(int) == 1) | (df["is_death_bowler"].astype(int) == 1)).astype(float)
    )
    df["flex_score"] = 0.55*(df["role"].isin(["AR", "WK"]).astype(float)) + 0.45*norm01(df["phase_versatility"])
    df["flex_score"] = norm01(df["flex_score"])

    # Auction strategy
    st.subheader("Auction Strategy")
    s1, s2, s3 = st.columns(3)
    strategy = s1.selectbox("Squad build style", ["Balanced", "Star-heavy", "Depth-heavy"], index=0)
    max_single_player_pct = s2.slider("Max single-player spend (% of remaining budget)", 10, 60, 30, 1)
    price_conc_penalty = s3.slider("Price concentration penalty", 0.0, 5.0, 1.0, 0.1)

    max_single_price = float(budget_after_ret) * (float(max_single_player_pct) / 100.0)

    # objective weights
    st.subheader("Multi-objective Weights")
    mw1, mw2, mw3, mw4, mw5, mw6 = st.columns(6)

    base_w_value = 1.0
    base_w_impact = 1.2
    base_w_fit = 1.0
    base_w_opp = 1.0
    base_w_flex = 0.8
    base_w_risk = 1.0

    if strategy == "Star-heavy":
        base_w_value = 0.8
        base_w_impact = 1.35
        base_w_fit = 1.15
        base_w_opp = 1.15
    elif strategy == "Depth-heavy":
        base_w_value = 1.25
        base_w_impact = 1.10
        price_conc_penalty = max(price_conc_penalty, 1.5)

    w_value = mw1.slider("Value gap", 0.0, 2.0, base_w_value, 0.05)
    w_impact = mw2.slider("Impact", 0.0, 2.0, base_w_impact, 0.05)
    w_fit = mw3.slider("Pitch fit", 0.0, 2.0, base_w_fit, 0.05)
    w_opp = mw4.slider("Opponent fit", 0.0, 2.0, base_w_opp, 0.05)
    w_flex = mw5.slider("Flex", 0.0, 2.0, base_w_flex, 0.05)
    w_risk = mw6.slider("Risk penalty", 0.0, 2.0, base_w_risk, 0.05)

    if season_goal in ["Batting dominance", "Bowling dominance"]:
        w_impact = max(w_impact, 1.3)

    df["value_gap"] = df["fair_salary_lakh"] - df["price_used_lakh"]
    df["value_gap_norm"] = norm01(df["value_gap"])
    df["impact_norm"] = norm01(df["match_impact_score"])
    df["fit_norm"] = norm01(df["pitch_fit"])
    df["opp_norm"] = norm01(df["opponent_fit"])
    df["risk_norm"] = norm01(df["total_risk"])

    df["objective_score_raw"] = (
        w_value*df["value_gap_norm"] +
        w_impact*df["impact_norm"] +
        w_fit*df["fit_norm"] +
        w_opp*df["opp_norm"] +
        w_flex*df["flex_score"] -
        w_risk*df["risk_norm"]
    )

    df["objective_score"] = df["objective_score_raw"] * df["risk_adjustment"]

    df["xi_score"] = (
        0.28*df["impact_norm"] +
        0.20*df["fit_norm"] +
        0.22*df["opp_norm"] +
        0.15*df["flex_score"] +
        0.15*df["value_gap_norm"]
    ) * df["risk_adjustment"]

    # Top table
    st.subheader("Top player table")
    st.dataframe(
        df[["player", "role", "age", "bat_hand", "batting_role", "bowling_role",
            "is_spinner", "is_pacer", "is_overseas",
            "price_used_lakh", "fair_salary_lakh", "value_gap",
            "match_impact_score", "pitch_fit", "opponent_fit",
            "flex_score", "total_risk", "objective_score"]]
        .sort_values("objective_score", ascending=False)
        .head(50),
        use_container_width=True
    )

    # constraint flags
    extra_min_flags = {
        "is_spinner": int(min_spinners),
        "is_pacer": int(min_pacers),
        "is_death_bowler2": int(min_death_bowl),
        "is_death_hitter": int(min_death_hit),
        "is_pp_bowler2": int(min_pp_bowl),
        "is_opener": int(min_openers),
        "is_finisher": int(min_finishers),
    }
    extra_max_flags = {"is_overseas": int(max_overseas_squad)}

    st.subheader("Optimized Squad Selection")

    squad, sm = optimize_squad_soft(
        df=df,
        budget_limit=float(budget_after_ret),
        max_players=int(max_players),
        min_role=min_role,
        price_col=price_col,
        extra_min_flags=extra_min_flags,
        extra_max_flags=extra_max_flags,
        lock_players=locked_set if locked_set else None,
        max_single_price=max_single_price,
        penalty_weight=float(penalty_weight),
        price_concentration_penalty=float(price_conc_penalty),
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Selected players", sm.get("count", 0))
    m2.metric("Spend", f"{sm.get('spend', 0):.1f} lakh")
    m3.metric("Total objective", f"{sm.get('objective', 0):.1f}")
    m4.metric("Overseas", f"{int((squad['is_overseas']==1).sum()) if len(squad) else 0}/{int(max_overseas_squad)}")
    m5.metric("Max player cap", f"{max_single_price:.0f} lakh")

    if sm.get("violations"):
        v = {k: round(v, 2) for k, v in sm["violations"].items() if v and v > 0.001}
        if v:
            st.warning("Soft-constraint shortfalls (mins missed by):")
            st.write(v)

    if len(squad) == 0:
        st.error("No squad returned. Budget/locks/caps might be impossible.")
        st.stop()

    st.dataframe(
        squad[["player","role","bat_hand","batting_role","bowling_role",
               "is_spinner","is_pacer","is_overseas",
               "price_used_lakh","fair_salary_lakh","value_gap",
               "match_impact_score","pitch_fit","opponent_fit","flex_score","total_risk","objective_score"]]
        .sort_values("objective_score", ascending=False),
        use_container_width=True
    )

    # Best XI
    st.subheader("Best Playing XI")
    x1, x2, x3, x4 = st.columns(4)
    xi_size = x1.number_input("XI size", value=11, step=1)
    xi_min_bat = x2.number_input("XI Min BAT", value=4, step=1)
    xi_min_bowl = x3.number_input("XI Min BOWL", value=4, step=1)
    xi_min_wk = x4.number_input("XI Min WK", value=1, step=1)

    x5, x6 = st.columns(2)
    xi_min_ar = x5.number_input("XI Min AR", value=1, step=1)
    max_overseas_xi = x6.number_input("XI Max Overseas", value=4, step=1)

    xi_min_role = {"BAT": xi_min_bat, "BOWL": xi_min_bowl, "AR": xi_min_ar, "WK": xi_min_wk}

    xi, xm = pick_best_xi(
        squad=squad,
        xi_size=int(xi_size),
        xi_min_role=xi_min_role,
        max_overseas_xi=int(max_overseas_xi),
        enforce_left_in_top4=enforce_left_in_top4
    )

    if len(xi) == 0:
        st.warning("No feasible XI under XI constraints. Relax XI mins / overseas / left-right constraint.")
    else:
        st.metric("XI total score", f"{xm.get('xi_score',0):.2f}")
        st.dataframe(
            xi[["player","role","bat_hand","batting_role","bowling_role",
                "is_spinner","is_pacer","is_overseas",
                "match_impact_score","pitch_fit","opponent_fit","flex_score","total_risk","xi_score"]]
            .sort_values("xi_score", ascending=False),
            use_container_width=True
        )

    # Downloads
    st.subheader("Downloads")
    st.download_button("Download full table (CSV)", to_csv_bytes(df), "auction_mode_full.csv", "text/csv")
    st.download_button("Download squad (CSV)", to_csv_bytes(squad), "auction_mode_squad.csv", "text/csv")
    st.download_button("Download XI (CSV)", to_csv_bytes(xi) if len(xi) else b"", "auction_mode_xi.csv", "text/csv")


# =========================================================
# HIGHLIGHTS GENERATOR (NEW 3rd MODE)
# =========================================================
def run_highlights_mode():
    st.subheader("Highlights Generator")
    st.caption("Upload a match video OR paste a YouTube link. Output: an MP4 clip you can download.")

    st.info("Starter behaviour: generates a demo highlight (first N seconds). Next step: plug in your real event-detection logic.")

    tab1, tab2 = st.tabs(["Upload Video", "YouTube Link"])
    video_path = None

    with tab1:
        vid = st.file_uploader("Upload video (mp4/mov/mkv)", type=["mp4", "mov", "mkv"], key="hl_upload")
        if vid is not None:
            tmp_dir = tempfile.mkdtemp()
            video_path = os.path.join(tmp_dir, vid.name)
            with open(video_path, "wb") as f:
                f.write(vid.getbuffer())
            st.success("Video uploaded ✅")
            st.video(video_path)

    with tab2:
        yt_url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...", key="hl_yt")
        if yt_url:
            st.warning("YouTube link requires yt-dlp + ffmpeg installed.")
            if st.button("Fetch from YouTube", key="hl_fetch"):
                tmp_dir = tempfile.mkdtemp()
                out_tpl = os.path.join(tmp_dir, "input.%(ext)s")
                cmd = ["yt-dlp", "-f", "bv*+ba/b", "-o", out_tpl, yt_url]
                try:
                    subprocess.check_call(cmd)
                    for fn in os.listdir(tmp_dir):
                        if fn.startswith("input."):
                            video_path = os.path.join(tmp_dir, fn)
                            break
                    if video_path and os.path.exists(video_path):
                        st.success("Downloaded ✅")
                        st.video(video_path)
                    else:
                        st.error("Download finished but file not found.")
                        return
                except Exception as e:
                    st.error(f"YouTube download failed: {e}")
                    return

    st.divider()
    st.subheader("Generate Highlights")

    c1, c2 = st.columns(2)
    clip_seconds = c1.number_input("Demo highlight length (seconds)", min_value=5, max_value=300, value=30, step=5)
    out_name = c2.text_input("Output filename", value="highlights.mp4")

    if st.button("Process", type="primary", key="hl_process"):
        if not video_path or not os.path.exists(video_path):
            st.error("Upload a video OR download via YouTube first.")
            return

        tmp_out_dir = tempfile.mkdtemp()
        out_path = os.path.join(tmp_out_dir, out_name)

        cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", "0", "-t", str(int(clip_seconds)), "-c", "copy", out_path]
        try:
            subprocess.check_call(cmd)
            st.success("Highlights generated ✅")
            st.video(out_path)

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download Highlights (MP4)",
                    data=f.read(),
                    file_name=out_name,
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Processing failed (ffmpeg issue): {e}")
            st.info("Fix: install ffmpeg and make sure it's in PATH.")


# =========================================================
# RUN SELECTED MODE  (UPDATED)
# =========================================================
if mode == "Scout Mode (Light)":
    run_scout_mode()
elif mode == "Auction Mode (Full)":
    run_auction_mode()
else:
    run_highlights_mode()
