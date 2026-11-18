"""Findings + ROI model shared by the notebook and the executive deck.

All benefit/cost assumptions below are ILLUSTRATIVE (the dataset is synthetic).
Replace them with company actuals before using the numbers for a real decision.
"""
import pandas as pd
from data_gen import make_data, kpis

# ---- Assumptions (edit here; the deck and notebook both read these) ----
MARGIN = 0.25            # contribution margin on incremental revenue
REALIZATION = 0.70       # haircut: share of the modelled benefit actually achieved
SHIFT = 0.30             # Rec 1: share of Paid Search + Social new customers moved to Email/Referral
CHURN_CUT_PT = 0.5       # Rec 2: monthly churn reduction in percentage points
RETAINED_MONTHS = 6      # Rec 2: extra months a saved customer keeps buying
AOV_LIFT = 0.03          # Rec 3: AOV lift on Electronics + Home & Living via bundles
INVEST = {"realloc": 8e5, "retention": 20e5, "bundles": 10e5}   # one-time, INR


def load() -> pd.DataFrame:
    d = make_data()
    d["year"] = d["month"].dt.year
    return d


def channel_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("channel").agg(rev=("revenue", "sum"), spend=("acq_spend", "sum"),
                                 new=("new_customers", "sum"))
    g["cac"] = g.spend / g.new
    g["rev_share"] = g.rev / g.rev.sum() * 100
    g["spend_share"] = g.spend / g.spend.sum() * 100
    return g


def roi(d: pd.DataFrame, realization: float = REALIZATION) -> dict:
    y = d[d.year == 2025]
    ch = y.groupby("channel").agg(spend=("acq_spend", "sum"), new=("new_customers", "sum"))
    paid, cheap = ch.loc[["Paid Search", "Social"]], ch.loc[["Email", "Referral"]]
    paid_cac, cheap_cac = paid.spend.sum() / paid.new.sum(), cheap.spend.sum() / cheap.new.sum()
    moved = SHIFT * paid.new.sum()
    r1_benefit = moved * (paid_cac - cheap_cac)

    churn_now = y.churned.sum() / y.start_customers.sum() * 100
    saved = y.start_customers.sum() * CHURN_CUT_PT / 100
    rev_per_cust_month = y.revenue.sum() / y.start_customers.sum()
    r2_rev = saved * rev_per_cust_month * RETAINED_MONTHS
    r2_benefit = r2_rev * MARGIN

    big = y[y.category.isin(["Electronics", "Home & Living"])].revenue.sum()
    r3_rev = big * AOV_LIFT
    r3_benefit = r3_rev * MARGIN

    out = {}
    for key, name, benefit, extra in [
        ("realloc", "Reallocate paid acquisition spend", r1_benefit, {"saving": r1_benefit}),
        ("retention", "Retention & win-back programme", r2_benefit, {"revenue": r2_rev}),
        ("bundles", "Electronics + Home bundles (AOV)", r3_benefit, {"revenue": r3_rev}),
    ]:
        b = benefit * realization
        out[key] = {"name": name, "invest": INVEST[key], "benefit": b,
                    "net": b - INVEST[key], "roi": (b - INVEST[key]) / INVEST[key] * 100,
                    "payback_months": INVEST[key] / (b / 12), **extra}
    tot_i = sum(v["invest"] for v in out.values())
    tot_b = sum(v["benefit"] for v in out.values())
    out["total"] = {"invest": tot_i, "benefit": tot_b, "net": tot_b - tot_i,
                    "roi": (tot_b - tot_i) / tot_i * 100, "payback_months": tot_i / (tot_b / 12)}
    out["detail"] = {"paid_cac": paid_cac, "cheap_cac": cheap_cac, "moved_customers": moved,
                     "churn_now": churn_now, "saved_customers": saved,
                     "rev_per_cust_month": rev_per_cust_month, "big_rev": big}
    return out


def findings(d: pd.DataFrame) -> dict:
    k24, k25 = kpis(d[d.year == 2024]), kpis(d[d.year == 2025])
    cat = d.groupby("category").revenue.sum().sort_values(ascending=False)
    reg = d.groupby("region").revenue.sum().sort_values(ascending=False)
    st = d.groupby("state").revenue.sum().sort_values(ascending=False)
    ch = channel_table(d)
    y25 = d[d.year == 2025]
    return {
        "k_all": kpis(d), "k24": k24, "k25": k25,
        "growth": (k25["revenue"] / k24["revenue"] - 1) * 100,
        "cat_share": cat / cat.sum() * 100, "reg_share": reg / reg.sum() * 100,
        "state_rev": st, "top4_share": st.head(4).sum() / st.sum() * 100,
        "channels": ch,
        "paid_spend_share": ch.loc[["Paid Search", "Social"], "spend"].sum() / ch.spend.sum() * 100,
        "paid_rev_share": ch.loc[["Paid Search", "Social"], "rev"].sum() / ch.rev.sum() * 100,
        "q4_share": y25[y25.month.dt.month >= 10].revenue.sum() / y25.revenue.sum() * 100,
        "total_spend": d.acq_spend.sum(),
    }
