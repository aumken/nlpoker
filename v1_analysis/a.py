import glob

import matplotlib.pyplot as plt
import pandas as pd

# 1. Load all log and summary CSV files
log_files = glob.glob("batch*_logs.csv")
logs = pd.concat([pd.read_csv(f) for f in log_files], ignore_index=True)

summary_files = [f for f in glob.glob("batch*.csv") if "_logs" not in f]
summary = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)

# 2. Preprocess logs: define streets
logs["board_cards"] = logs["board_cards"].fillna("").astype(str)
logs["board_card_count"] = logs["board_cards"].apply(
    lambda x: len(x.split()) if x.strip() else 0
)
logs["street"] = (
    logs["board_card_count"]
    .map({0: "preflop", 3: "flop", 4: "turn", 5: "river"})
    .fillna("other")
)

# 3. Calculate hand-level metrics (VPIP, PFR, WTSD, WSD, Win Rate)
hand_groups = logs.groupby(["simulation_index", "round_id", "player_id", "model_name"])


def compute_hand_metrics(g):
    preflop = g[g["street"] == "preflop"]
    vpip = (
        (preflop["action"] == "CALL") | preflop["action"].str.startswith("RAISE")
    ).any()
    pfr = preflop["action"].str.startswith("RAISE").any()
    saw_river = (g["street"] == "river").any()
    won = (g["action"] == "WINNER").any()
    return pd.Series({"vpip": vpip, "pfr": pfr, "saw_river": saw_river, "won": won})


hand_metrics = hand_groups.apply(compute_hand_metrics).reset_index()
model_metrics = (
    hand_metrics.groupby("model_name")
    .agg(
        VPIP=("vpip", "mean"),
        PFR=("pfr", "mean"),
        WTSD=("saw_river", "mean"),
    )
    .reset_index()
)

wssd = (
    hand_metrics[hand_metrics["saw_river"]]
    .groupby("model_name")["won"]
    .mean()
    .reset_index(name="WSD")
)
win_rate = hand_metrics.groupby("model_name")["won"].mean().reset_index(name="Win_Rate")
model_metrics = model_metrics.merge(wssd, on="model_name").merge(
    win_rate, on="model_name"
)

print("\nModel Poker Analytics Summary:")
print(model_metrics)

# Plot core metrics
for metric in ["VPIP", "PFR", "WTSD", "WSD", "Win_Rate"]:
    plt.figure()
    plt.bar(model_metrics["model_name"], model_metrics[metric])
    plt.title(f"{metric} by Model")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4. Aggression Factor overall (raises / calls)
action_counts = (
    logs.groupby("model_name")["action"].value_counts().unstack(fill_value=0)
)
action_counts["n_raises"] = action_counts.filter(like="RAISE").sum(axis=1)
action_counts["n_calls"] = action_counts["CALL"]
action_counts["Aggression_Factor"] = action_counts["n_raises"] / action_counts[
    "n_calls"
].replace(0, pd.NA)
af = action_counts[["Aggression_Factor"]].reset_index()

print("\nAggression Factor by Model:")
print(af)
plt.figure()
plt.bar(af["model_name"], af["Aggression_Factor"])
plt.title("Aggression Factor by Model")
plt.xlabel("Model")
plt.ylabel("Aggression Factor")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Stack Evolution: average ending stack per simulation
winners = logs[logs["action"] == "WINNER"]
stack_evo = (
    winners.groupby(["model_name", "simulation_index"])["ending_stack"]
    .mean()
    .unstack(level=0)
)
plt.figure()
for model in stack_evo.columns:
    plt.plot(stack_evo.index, stack_evo[model].fillna(0), marker="o", label=model)
plt.title("Average Ending Stack by Simulation")
plt.xlabel("Simulation Index")
plt.ylabel("Average Ending Stack")
plt.legend()
plt.tight_layout()
plt.show()

# 6. Action frequency by position & action
pos_action = (
    logs.groupby(["model_name", "player_position", "action"])
    .size()
    .reset_index(name="count")
)
pos_totals = (
    pos_action.groupby(["model_name", "player_position"])["count"]
    .sum()
    .reset_index(name="total")
)
pos_action = pos_action.merge(pos_totals, on=["model_name", "player_position"])
pos_action["frequency"] = pos_action["count"] / pos_action["total"]
pivot = pos_action.pivot_table(
    index=["model_name", "action"],
    columns="player_position",
    values="frequency",
    fill_value=0,
)
print("\nAction Frequency by Position and Action:")
print(pivot)

# 7. Aggression Factor drift over simulations
sim_counts = (
    logs.groupby(["model_name", "simulation_index"])["action"]
    .value_counts()
    .unstack(fill_value=0)
)
sim_counts["raises"] = sim_counts.filter(like="RAISE").sum(axis=1)
sim_counts["calls"] = sim_counts["CALL"]
sim_counts["AF"] = sim_counts["raises"] / sim_counts["calls"].replace(0, pd.NA)
af_sim = sim_counts["AF"].unstack(level=0).fillna(0)

plt.figure()
for model in af_sim.columns:
    plt.plot(af_sim.index, af_sim[model], marker="x", label=model)
plt.title("Aggression Factor by Simulation")
plt.xlabel("Simulation Index")
plt.ylabel("Aggression Factor")
plt.legend()
plt.tight_layout()
plt.show()
