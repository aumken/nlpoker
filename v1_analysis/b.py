import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_logs(pattern="batch*_logs.csv"):
    """
    Load all batch log CSVs into a DataFrame, add 'street' and 'raise_amount' fields.
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No log files matching {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["board_cards"] = df["board_cards"].fillna("").astype(str)
    df["board_card_count"] = df["board_cards"].apply(
        lambda x: len(x.split()) if x.strip() else 0
    )
    df["street"] = (
        df["board_card_count"]
        .map({0: "preflop", 3: "flop", 4: "turn", 5: "river"})
        .fillna("other")
    )
    df["raise_amount"] = (
        df["action"].str.extract(r"RAISE_(\d+)").astype(float).fillna(0)
    )
    return df


def compute_preflop_metrics(logs):
    """
    VPIP: Voluntarily Put Money In Pot %
      - How often the model CALLs or RAISEs pre-flop (loose vs. tight)
      - Calculation: (# hands with CALL or RAISE pre-flop) / (total hands dealt)
    PFR: Pre-Flop Raise %
      - How often the model RAISEs pre-flop
      - Calculation: (# hands with RAISE pre-flop) / (total hands dealt)
    Returns DataFrame: model_name, VPIP, PFR
    """
    pf = logs[logs["street"] == "preflop"]
    hand_groups = pf.groupby(
        ["simulation_index", "round_id", "player_id", "model_name"]
    )

    def hand_metrics(g):
        actions = g["action"]
        vpip = ((actions == "CALL") | actions.str.startswith("RAISE")).any()
        pfr = actions.str.startswith("RAISE").any()
        return pd.Series({"vpip": vpip, "pfr": pfr})

    hm = hand_groups.apply(hand_metrics).reset_index()
    summary = hm.groupby("model_name")[["vpip", "pfr"]].mean().reset_index()
    summary.rename(columns={"vpip": "VPIP", "pfr": "PFR"}, inplace=True)
    return summary


def compute_postflop_metrics(logs):
    """
    AFq: Aggression Frequency %
      - (RAISEs) / (RAISE + CALL + CHECK) after flop
    WTSD: Went To Showdown %
      - (hands saw river) / (hands saw flop)
    WSD: Won at Showdown %
      - (showdowns won) / (showdowns reached)
    Returns DataFrame: model_name, AFq, WTSD, WSD
    """
    hand_groups = logs.groupby(
        ["simulation_index", "round_id", "player_id", "model_name"]
    )

    def pf_metrics(g):
        post = g[g["street"].isin(["flop", "turn", "river"])]
        raises = post["action"].str.startswith("RAISE").sum()
        calls_checks = post["action"].isin(["CALL", "CHECK"]).sum()
        afq = raises / (raises + calls_checks) if (raises + calls_checks) > 0 else 0
        saw_flop = g["street"].eq("flop").any()
        saw_river = g["street"].eq("river").any()
        wtsd = saw_river / saw_flop if saw_flop else 0
        won = g["action"].eq("WINNER").any()
        wsd = won / saw_river if saw_river else 0
        return pd.Series({"AFq": afq, "WTSD": wtsd, "WSD": wsd})

    m = hand_groups.apply(pf_metrics).reset_index()
    summary = m.groupby("model_name")[["AFq", "WTSD", "WSD"]].mean().reset_index()
    return summary


def compute_bet_sizing(logs, big_blind=10):
    """
    PF_Raise_xBB: Avg pre-flop raise size relative to big blind
    Avg_Pct_Pot_Raise: Avg post-flop raise as % of pot
    Returns DataFrame: model_name, PF_Raise_xBB, Avg_Pct_Pot_Raise
    """
    # Pre-flop
    pf = logs[logs["street"] == "preflop"]
    pf_r = pf[pf["raise_amount"] > 0]
    pf_sum = pf_r.groupby("model_name")["raise_amount"].mean().reset_index()
    pf_sum["PF_Raise_xBB"] = pf_sum["raise_amount"] / big_blind
    # Post-flop
    post = logs[logs["street"].isin(["flop", "turn", "river"])]
    post_r = post[post["raise_amount"] > 0].copy()
    post_r["pot_before"] = post_r["pot_size"] - post_r["raise_amount"]
    post_r["Pct_Pot"] = post_r["raise_amount"] / post_r["pot_before"].replace(0, pd.NA)
    post_sum = (
        post_r.groupby("model_name")["Pct_Pot"]
        .mean()
        .reset_index()
        .rename(columns={"Pct_Pot": "Avg_Pct_Pot_Raise"})
    )
    return pf_sum.merge(post_sum, on="model_name", how="outer").fillna(0)[
        ["model_name", "PF_Raise_xBB", "Avg_Pct_Pot_Raise"]
    ]


def compute_positional_metrics(logs):
    """
    VPIP/PFR by Position (Early POS1 vs Late POS3)
    Returns pivoted DataFrame with columns: model_name, vpip_POS1, pfr_POS1, vpip_POS3, pfr_POS3
    """
    pf = logs[logs["street"] == "preflop"]
    grp = pf.groupby(
        ["simulation_index", "round_id", "player_id", "model_name", "player_position"]
    )

    def flags(g):
        acts = g["action"]
        vpip = ((acts == "CALL") | acts.str.startswith("RAISE")).any()
        pfr = acts.str.startswith("RAISE").any()
        return pd.Series({"vpip": vpip, "pfr": pfr})

    df = grp.apply(flags).reset_index()
    summ = (
        df.groupby(["model_name", "player_position"])[["vpip", "pfr"]]
        .mean()
        .reset_index()
    )
    piv = summ.pivot(
        index="model_name", columns="player_position", values=["vpip", "pfr"]
    )
    piv.columns = [f"{m}_POS{p}" for m, p in piv.columns]
    return piv.reset_index()


def plot_bar(df, x, ys, title, ylabel, labels=None, filename=None):
    """Generic grouped bar chart. Saves if filename is provided."""
    plt.figure() # Create a new figure for each plot
    ind = range(len(df))
    n = len(ys)
    width = 0.8 / n
    
    # Create short model names by splitting at '/' and taking first part
    short_names = df[x].apply(lambda s: s.split('/')[0])
    
    for i, col in enumerate(ys):
        plt.bar(
            [p + i * width for p in ind],
            df[col],
            width=width,
            label=(labels[i] if labels else col),
        )
    plt.xticks([p + width * (n - 1) / 2 for p in ind], short_names, rotation=45)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    
    if filename:
        # Ensure plots directory exists
        dir_name = os.path.dirname(filename)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(filename)
        plt.close() # Close the figure after saving
    else:
        plt.show()


def main():
    # Ensure plots directory exists at the start
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    logs = load_logs()
    pre = compute_preflop_metrics(logs)
    post = compute_postflop_metrics(logs)
    bet = compute_bet_sizing(logs)
    pos = compute_positional_metrics(logs)

    print("\nPre-flop Metrics (VPIP, PFR):")
    print(pre)
    plot_bar(
        pre,
        "model_name",
        ["VPIP", "PFR"],
        "Pre-flop VPIP & PFR by Model",
        "%",
        labels=["VPIP", "PFR"],
        filename="plots/preflop_metrics.png",
    )

    print("\nPost-flop Metrics (AFq, WTSD, WSD):")
    print(post)
    plot_bar(
        post,
        "model_name",
        ["AFq", "WTSD", "WSD"],
        "Post-flop Metrics by Model",
        "%",
        labels=["AFq", "WTSD", "WSD"],
        filename="plots/postflop_metrics.png",
    )

    print("\nBet Sizing Insights:")
    print(bet)
    plot_bar(
        bet,
        "model_name",
        ["PF_Raise_xBB", "Avg_Pct_Pot_Raise"],
        "Bet Sizing by Model",
        "Value",
        labels=["Avg Pre-flop Raise (xBB)", "Avg Post-flop Raise (% Pot)"],
        filename="plots/bet_sizing.png",
    )

    print("\nPositional VPIP/PFR (POS1 vs POS3):")
    print(pos)
    # Combine VPIP and PFR positional plots into one
    plot_bar(
        pos,
        "model_name",
        ["vpip_POS1", "pfr_POS1", "vpip_POS3", "pfr_POS3"],
        "Positional VPIP & PFR by Model",
        "%",
        labels=[
            "VPIP Early (POS1)",
            "PFR Early (POS1)",
            "VPIP Late (POS3)",
            "PFR Late (POS3)",
        ],
        filename="plots/positional_metrics.png",
    )


if __name__ == "__main__":
    main()
