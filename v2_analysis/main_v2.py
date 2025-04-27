# -*- coding: utf-8 -*-
"""
Analyzes poker simulation logs with a new directory structure and CSV format.

Reads logs from folders like 'simulation_X.Y/', processes data to calculate
poker statistics (VPIP, PFR, AFq, WTSD, WSD, Bet Sizing, Positional)
grouped by AI model AND temperature, prints results, and generates plots:
 - Bar plots showing metric trends vs temperature.
 - Subplots showing chip stack evolution per game, colored by model.
"""

import glob
import math
import os
import re
import traceback
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==============================================================================
# Constants
# ==============================================================================
OUTPUT_PLOT_DIR = "plots"  # Output directory name
BIG_BLIND = 10
STARTING_STACK = 1000
# Define consistent colors and markers for models across plots
MODEL_VISUALS = {
    "Meta-Llama-3.1-8B-Instruct-Turbo": {"color": "blue", "marker": "o"},
    "Mistral-7B-Instruct-v0.3": {"color": "green", "marker": "s"},
    "Qwen2.5-7B-Instruct-Turbo": {"color": "red", "marker": "^"},
    # Add more models here if needed, cycle colors/markers if list exceeds definitions
}
DEFAULT_VISUAL = {"color": "grey", "marker": "x"}

# Define consistent colors for base models for chip stack plot
MODEL_COLORS = {
    "Meta-Llama-3.1-8B-Instruct-Turbo": "blue",
    "Mistral-7B-Instruct-v0.3": "green",
    "Qwen2.5-7B-Instruct-Turbo": "red",
    "UnknownModel": "grey",  # Fallback
}
DEFAULT_MODEL_COLOR = "purple"  # If a model is not in the dict


# ==============================================================================
# Data Loading and Preparation (Unchanged)
# ==============================================================================
def load_all_logs(base_pattern="simulation_*", file_pattern="game_*.csv"):
    all_files = []
    simulation_folders = glob.glob(base_pattern)
    if not simulation_folders:
        raise FileNotFoundError(
            f"No simulation folders found matching '{base_pattern}'"
        )
    print(f"Found simulation folders: {simulation_folders}")
    for folder in simulation_folders:
        temp_match = re.search(r"simulation_(\d+(\.\d+)?)", folder)
        folder_temp = float(temp_match.group(1)) if temp_match else None
        path = Path(folder)
        files_in_folder = list(path.glob(file_pattern))
        print(
            f"  - Folder '{folder}': Found {len(files_in_folder)} files matching '{file_pattern}'. Temp from folder: {folder_temp}"
        )
        all_files.extend([(f, folder_temp) for f in files_in_folder])
    if not all_files:
        raise FileNotFoundError(
            f"No log files found matching '{file_pattern}' within folders like '{base_pattern}'"
        )
    print(f"\nLoading {len(all_files)} total CSV files...")
    df_list = []
    for f, folder_temp in all_files:
        try:
            df_single = pd.read_csv(f)
            if "Temperature" not in df_single.columns:
                df_single["Temperature"] = folder_temp
            else:
                df_single["Temperature"] = (
                    df_single.groupby(["GameID", "RoundID"])["Temperature"]
                    .ffill()
                    .bfill()
                )
                if folder_temp is not None:
                    df_single["Temperature"].fillna(folder_temp, inplace=True)
            df_list.append(df_single)
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file: {f}")
        except Exception as e:
            print(f"Error reading file {f}: {e}")
    if not df_list:
        raise ValueError("No valid data could be loaded.")
    print("Concatenating DataFrames...")
    logs = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(logs)} total actions.")
    logs["BoardCards"] = logs["BoardCards"].fillna("").astype(str)
    logs["Temperature"] = pd.to_numeric(logs["Temperature"], errors="coerce").fillna(
        0.0
    )
    logs["ActionAmount"] = pd.to_numeric(logs["ActionAmount"], errors="coerce").fillna(
        0
    )
    logs["AmountWon"] = pd.to_numeric(logs["AmountWon"], errors="coerce").fillna(0)
    logs["PotTotalBefore"] = pd.to_numeric(
        logs["PotTotalBefore"], errors="coerce"
    ).fillna(0)
    logs["ModelShortName"] = (
        logs["ModelName"].str.split("/").str[-1].fillna("UnknownModel")
    )
    logs["model_temp_id"] = (
        logs["ModelShortName"] + "@" + logs["Temperature"].astype(str)
    )
    logs["Position"] = logs["Position"].astype(str).str.upper().str.strip()
    logs.loc[logs["Position"] == "BTN/SB", "Position"] = "BTN_HU"
    print("Data loading and preparation complete.")
    print(f"Unique Model/Temp combinations found: {logs['model_temp_id'].nunique()}")
    return logs


# ==============================================================================
# Confidence Interval Calculation
# ==============================================================================
def compute_confidence_interval(data, confidence=0.95):
    """
    Computes confidence interval for given data.
    
    Args:
        data: Series or array of values to compute CI for
        confidence: confidence level (default 0.95 for 95% CI)
    
    Returns:
        float: margin of error or 0 if data insufficient
    """
    if len(data) < 2:
        return 0  # No error bar if insufficient data
        
    # Calculate standard error and margin of error
    a = 1.0 * np.array(data)
    n = len(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h  # Return the margin of error


# ==============================================================================
# Metric Calculation Functions (With Raw Data For CI)
# ==============================================================================
def compute_preflop_metrics_with_ci(logs):
    pf = logs[(logs["Street"] == "Preflop") & logs["PlayerID"].notna()].copy()
    dealt_hands = pf[pf["ActionType"] == "DEALT_HOLE"][
        [
            "GameID",
            "RoundID",
            "PlayerID",
            "model_temp_id",
            "ModelShortName",
            "Temperature",
        ]
    ].drop_duplicates()
    voluntary_actions = pf[pf["ActionType"].isin(["CALL", "BET", "RAISE"])]
    vpip_hands = voluntary_actions[["GameID", "RoundID", "PlayerID"]].drop_duplicates()
    vpip_hands["vpip_flag"] = True
    pfr_actions = pf[pf["ActionType"].isin(["BET", "RAISE"])]
    pfr_hands = pfr_actions[["GameID", "RoundID", "PlayerID"]].drop_duplicates()
    pfr_hands["pfr_flag"] = True
    summary = dealt_hands.merge(
        vpip_hands, on=["GameID", "RoundID", "PlayerID"], how="left"
    )
    summary = summary.merge(pfr_hands, on=["GameID", "RoundID", "PlayerID"], how="left")
    summary[["vpip_flag", "pfr_flag"]] = summary[["vpip_flag", "pfr_flag"]].fillna(
        False
    )
    # Convert bool to int for easier calculation
    summary["vpip_flag"] = summary["vpip_flag"].astype(int)
    summary["pfr_flag"] = summary["pfr_flag"].astype(int)
    
    group_cols = ["model_temp_id", "ModelShortName", "Temperature"]
    # Calculate mean metrics
    final_summary = (
        summary.groupby(group_cols)[["vpip_flag", "pfr_flag"]].mean().reset_index()
    )
    final_summary.rename(columns={"vpip_flag": "VPIP", "pfr_flag": "PFR"}, inplace=True)
    
    # Calculate confidence intervals
    ci_data = {}
    for model in summary["ModelShortName"].unique():
        for temp in summary["Temperature"].unique():
            model_temp_data = summary[(summary["ModelShortName"] == model) & 
                                      (summary["Temperature"] == temp)]
            if not model_temp_data.empty:
                vpip_ci = compute_confidence_interval(model_temp_data["vpip_flag"])
                pfr_ci = compute_confidence_interval(model_temp_data["pfr_flag"])
                ci_data[(model, temp, "VPIP")] = vpip_ci
                ci_data[(model, temp, "PFR")] = pfr_ci
    
    return final_summary, ci_data


def compute_postflop_metrics_with_ci(logs):
    player_logs = logs[logs["PlayerID"].notna()].copy()
    group_cols = [
        "GameID",
        "RoundID",
        "PlayerID",
        "model_temp_id",
        "ModelShortName",
        "Temperature",
    ]
    hand_groups = player_logs.groupby(group_cols)

    def hand_postflop_metrics(g):
        saw_flop = "Flop" in g["Street"].values
        saw_river = "River" in g["Street"].values
        postflop_actions = g[g["Street"].isin(["Flop", "Turn", "River"])]
        bets_raises = postflop_actions["ActionType"].isin(["BET", "RAISE"]).sum()
        total_actions = (
            postflop_actions["ActionType"].isin(["BET", "RAISE", "CALL", "CHECK"]).sum()
        )
        afq = np.divide(bets_raises, total_actions) if total_actions > 0 else 0
        wtsd = float(saw_river) if saw_flop else np.nan
        reached_showdown = saw_river
        won_at_showdown = False
        if reached_showdown:
            win_action = g[(g["ActionType"] == "WINS_POT") & (g["AmountWon"] > 0)]
            if not win_action.empty:
                river_action_ids = g[g["Street"] == "River"]["ActionID"]
                if not river_action_ids.empty:
                    first_river_action_id = river_action_ids.min()
                    if (win_action["ActionID"] >= first_river_action_id).any():
                        relevant_wins = win_action[
                            win_action["ActionID"] >= first_river_action_id
                        ]
                        if "FinalHandRank" in relevant_wins.columns:
                            if not (
                                relevant_wins["FinalHandRank"] == "Premature Win"
                            ).all():
                                won_at_showdown = True
                        else:
                            won_at_showdown = True
        wsd = float(won_at_showdown) if reached_showdown else np.nan
        return pd.Series({"AFq": afq, "WTSD": wtsd, "WSD": wsd})

    metrics = hand_groups.apply(hand_postflop_metrics)
    metrics = metrics.reset_index()
    agg_group_cols = ["model_temp_id", "ModelShortName", "Temperature"]
    summary = (
        metrics.groupby(agg_group_cols)[["AFq", "WTSD", "WSD"]].mean().reset_index()
    )
    summary[["AFq", "WTSD", "WSD"]] = summary[["AFq", "WTSD", "WSD"]].fillna(0.0)
    
    # Calculate confidence intervals
    ci_data = {}
    for model in metrics["ModelShortName"].unique():
        for temp in metrics["Temperature"].unique():
            model_temp_data = metrics[(metrics["ModelShortName"] == model) & 
                                      (metrics["Temperature"] == temp)]
            
            if not model_temp_data.empty:
                for metric in ["AFq", "WTSD", "WSD"]:
                    # Filter out NaNs for correct CI calculation
                    metric_data = model_temp_data[metric].dropna()
                    if len(metric_data) >= 2:  # Need at least 2 data points
                        ci_data[(model, temp, metric)] = compute_confidence_interval(metric_data)
                    else:
                        ci_data[(model, temp, metric)] = 0
    
    return summary, ci_data


def compute_bet_sizing_with_ci(logs, big_blind=BIG_BLIND):
    player_logs = logs[logs["PlayerID"].notna()].copy()
    
    # Preflop raise data
    pf_aggr = player_logs[
        (player_logs["Street"] == "Preflop")
        & (player_logs["ActionType"].isin(["BET", "RAISE"]))
        & (player_logs["ActionAmount"] > 0)
    ].copy()
    
    # Postflop data
    postf_aggr = player_logs[
        player_logs["Street"].isin(["Flop", "Turn", "River"])
        & player_logs["ActionType"].isin(["BET", "RAISE"])
        & (player_logs["ActionAmount"] > 0)
    ].copy()
    postf_aggr["Pct_Pot"] = np.divide(
        postf_aggr["ActionAmount"], postf_aggr["PotTotalBefore"]
    )
    postf_aggr["Pct_Pot"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Group columns
    group_cols = ["model_temp_id", "ModelShortName", "Temperature"]
    
    # Calculate means
    pf_avg_raise = pf_aggr.groupby(group_cols)["ActionAmount"].mean().reset_index()
    pf_avg_raise.rename(columns={"ActionAmount": "Avg_PF_Raise_Amount"}, inplace=True)
    pf_avg_raise["PF_Raise_xBB"] = pf_avg_raise["Avg_PF_Raise_Amount"] / big_blind
    
    postf_avg_pct_pot = postf_aggr.groupby(group_cols)["Pct_Pot"].mean().reset_index()
    postf_avg_pct_pot.rename(columns={"Pct_Pot": "Avg_Pct_Pot_Raise"}, inplace=True)
    
    # Merge results
    merge_on_cols = ["model_temp_id", "ModelShortName", "Temperature"]
    summary = pf_avg_raise.merge(postf_avg_pct_pot, on=merge_on_cols, how="outer")
    
    final_cols = merge_on_cols + ["PF_Raise_xBB", "Avg_Pct_Pot_Raise"]
    for col in final_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    summary = summary[final_cols]
    summary[["PF_Raise_xBB", "Avg_Pct_Pot_Raise"]] = summary[
        ["PF_Raise_xBB", "Avg_Pct_Pot_Raise"]
    ].fillna(0)
    
    # Calculate confidence intervals
    ci_data = {}
    
    # Preflop CI
    for model in pf_aggr["ModelShortName"].unique():
        for temp in pf_aggr["Temperature"].unique():
            model_temp_data = pf_aggr[(pf_aggr["ModelShortName"] == model) & 
                                      (pf_aggr["Temperature"] == temp)]
            if not model_temp_data.empty:
                pf_bb_amounts = model_temp_data["ActionAmount"] / big_blind
                ci_data[(model, temp, "PF_Raise_xBB")] = compute_confidence_interval(pf_bb_amounts)
    
    # Postflop CI
    for model in postf_aggr["ModelShortName"].unique():
        for temp in postf_aggr["Temperature"].unique():
            model_temp_data = postf_aggr[(postf_aggr["ModelShortName"] == model) & 
                                        (postf_aggr["Temperature"] == temp)]
            if not model_temp_data.empty:
                pct_pot_values = model_temp_data["Pct_Pot"].dropna()  # Remove NaNs
                if len(pct_pot_values) >= 2:
                    ci_data[(model, temp, "Avg_Pct_Pot_Raise")] = compute_confidence_interval(pct_pot_values)
                else:
                    ci_data[(model, temp, "Avg_Pct_Pot_Raise")] = 0
    
    return summary, ci_data


def compute_positional_metrics_with_ci(logs):
    pf = logs[
        (logs["Street"] == "Preflop")
        & logs["PlayerID"].notna()
        & logs["Position"].notna()
        & (logs["Position"] != "NAN")
    ].copy()
    
    # Extract hands by position
    dealt_hands_pos = pf[
        [
            "GameID",
            "RoundID",
            "PlayerID",
            "model_temp_id",
            "ModelShortName",
            "Temperature",
            "Position",
        ]
    ].drop_duplicates()
    
    # Get voluntary actions
    voluntary_actions = pf[pf["ActionType"].isin(["CALL", "BET", "RAISE"])]
    vpip_hands = voluntary_actions[["GameID", "RoundID", "PlayerID"]].drop_duplicates()
    vpip_hands["vpip_flag"] = True
    
    # Get raises
    pfr_actions = pf[pf["ActionType"].isin(["BET", "RAISE"])]
    pfr_hands = pfr_actions[["GameID", "RoundID", "PlayerID"]].drop_duplicates()
    pfr_hands["pfr_flag"] = True
    
    # Merge flags
    summary = dealt_hands_pos.merge(
        vpip_hands, on=["GameID", "RoundID", "PlayerID"], how="left"
    )
    summary = summary.merge(pfr_hands, on=["GameID", "RoundID", "PlayerID"], how="left")
    summary[["vpip_flag", "pfr_flag"]] = summary[["vpip_flag", "pfr_flag"]].fillna(
        False
    )
    # Convert bool to int for calculation
    summary["vpip_flag"] = summary["vpip_flag"].astype(int)
    summary["pfr_flag"] = summary["pfr_flag"].astype(int)
    
    # Calculate means
    group_cols = ["model_temp_id", "ModelShortName", "Temperature", "Position"]
    agg_summary = (
        summary.groupby(group_cols)[["vpip_flag", "pfr_flag"]].mean().reset_index()
    )
    
    # Store raw data for CI calculation later
    raw_data = summary.copy()
    
    # Pivot the data
    try:
        pivot_df = agg_summary.pivot(
            index=["model_temp_id", "ModelShortName", "Temperature"],
            columns="Position",
            values=["vpip_flag", "pfr_flag"],
        )
        pivot_df.columns = [
            f"{metric.replace('_flag','')}_{pos}" for metric, pos in pivot_df.columns
        ]
        pivot_df.reset_index(inplace=True)
    except Exception as e:
        print(
            f"Error pivoting positional data: {e}\nAggregation summary before pivot:\n{agg_summary}"
        )
        return agg_summary.fillna(0), {}
    
    # Calculate confidence intervals
    ci_data = {}
    for model in raw_data["ModelShortName"].unique():
        for temp in raw_data["Temperature"].unique():
            for pos in raw_data["Position"].unique():
                # Filter data for this model, temperature, position
                pos_data = raw_data[(raw_data["ModelShortName"] == model) & 
                                   (raw_data["Temperature"] == temp) &
                                   (raw_data["Position"] == pos)]
                
                if not pos_data.empty:
                    vpip_ci = compute_confidence_interval(pos_data["vpip_flag"])
                    pfr_ci = compute_confidence_interval(pos_data["pfr_flag"])
                    
                    ci_data[(model, temp, f"vpip_{pos}")] = vpip_ci
                    ci_data[(model, temp, f"pfr_{pos}")] = pfr_ci
    
    return pivot_df.fillna(0), ci_data


# ==============================================================================
# Plotting Function (Bar Plots vs Temperature) (With Confidence Intervals)
# ==============================================================================
def plot_metric_bars_vs_temp(
    df, metrics_to_plot, fig_title, filename=None, sharey=False, ci_data=None
):
    if df.empty:
        print(f"Skipping plot '{fig_title}' - DataFrame is empty.")
        return
    required_cols = ["ModelShortName", "Temperature"] + metrics_to_plot
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping plot '{fig_title}' - DataFrame missing required columns: {missing}"
        )
        return

    models = sorted(df["ModelShortName"].unique())
    temps = sorted(df["Temperature"].unique())
    n_metrics = len(metrics_to_plot)
    ncols = min(n_metrics, 3)
    nrows = math.ceil(n_metrics / ncols)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except IOError:
        try:
            plt.style.use("seaborn-whitegrid")
        except IOError:
            print("Warning: Seaborn styles not found. Using default style.")

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5), sharey=sharey, squeeze=False
    )
    axs = axs.flatten()

    # Create color mapping for consistent colors across subplots
    n_models = len(models)
    bar_width = 0.8 / n_models  # Width of bars, adjusted for number of models

    # Legend elements
    legend_handles = []

    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]

        # For each temperature value
        for temp_idx, temp in enumerate(temps):
            temp_data = df[df["Temperature"] == temp]

            # For each model at this temperature
            for model_idx, model in enumerate(models):
                model_data = temp_data[temp_data["ModelShortName"] == model]

                if not model_data.empty:
                    # Calculate position for this model's bar
                    x_pos = temp_idx + (model_idx - n_models / 2 + 0.5) * bar_width

                    # Get color for this model
                    visuals = MODEL_VISUALS.get(model, DEFAULT_VISUAL)
                    color = visuals["color"]

                    # Get metric value
                    value = model_data[metric].values[0]
                    
                    # Get error margin if available
                    error = 0
                    if ci_data is not None:
                        key = (model, temp, metric)
                        if key in ci_data:
                            error = ci_data[key]

                    # Plot bar with error bar
                    bar = ax.bar(
                        x_pos,
                        value,
                        width=bar_width,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        yerr=error,  # Add error bar
                        capsize=4,   # Cap width for error bar
                        label=model if i == 0 and temp_idx == 0 else "_nolegend_",
                    )

                    # Add to legend only once per model
                    if i == 0 and temp_idx == 0:
                        legend_handles.append(Patch(color=color, label=model))

        ax.set_title(metric, fontsize=13, fontweight="medium")
        ax.set_xlabel("Temperature", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels(temps)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=1.03)

    # Add legend
    fig.legend(
        handles=legend_handles,
        title="Models",
        loc="upper right",
        bbox_to_anchor=(1.18, 0.98),
        fontsize=10,
        title_fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 0.85, 0.98])

    if filename:
        dir_name = os.path.dirname(filename)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
        plt.savefig(filename, bbox_inches="tight", dpi=120)
        print(f"Plot saved to: {filename}")
        plt.close(fig)
    else:
        plt.show()


# ==============================================================================
# Chip Stack Data Extraction (REVISED with forward fill)
# ==============================================================================
def extract_round_start_stacks(logs, starting_stack=STARTING_STACK):
    """
    Extracts the stack size for each player profile at the start of each round
    for every game they played. Includes the initial stack at round 0 and
    forward-fills stacks for rounds where the player didn't act.
    """
    print("Extracting start-of-round stacks (with forward fill)...")

    # 1. Get stacks from the first action per player/round/game where available
    first_action_indices = logs.loc[
        logs.groupby(["GameID", "RoundID", "PlayerID"])["ActionID"].idxmin()
    ].index
    first_action_per_round = logs.loc[first_action_indices]
    stack_data_raw = first_action_per_round[
        [
            "GameID",
            "RoundID",
            "PlayerID",
            "model_temp_id",
            "StackBefore",
            "ModelShortName",
            "Temperature",
        ]
    ].copy()

    # 2. Create initial stack data (Round 0) for every player in every game
    game_players = logs[
        ["GameID", "PlayerID", "model_temp_id", "ModelShortName", "Temperature"]
    ].drop_duplicates(subset=["GameID", "PlayerID"])
    initial_stacks = game_players.copy()
    initial_stacks["RoundID"] = 0
    initial_stacks["StackBefore"] = starting_stack

    # 3. Combine initial stacks with observed stacks
    combined_stacks = pd.concat([initial_stacks, stack_data_raw], ignore_index=True)

    # 4. Determine the full range of rounds for each game/player combination
    # Max round per game
    max_rounds = logs.groupby("GameID")["RoundID"].max()
    # All participating players per game
    participants = combined_stacks[
        ["GameID", "PlayerID", "model_temp_id", "ModelShortName", "Temperature"]
    ].drop_duplicates()

    # Create a full index grid
    full_index_list = []
    for _, row in participants.iterrows():
        game_id = row["GameID"]
        max_round_for_game = max_rounds.get(
            game_id, 0
        )  # Get max round, default to 0 if game not found
        rounds = range(max_round_for_game + 1)  # Rounds from 0 to max_round
        for round_id in rounds:
            full_index_list.append(
                {
                    "GameID": game_id,
                    "RoundID": round_id,
                    "PlayerID": row["PlayerID"],
                    "model_temp_id": row["model_temp_id"],
                    "ModelShortName": row["ModelShortName"],
                    "Temperature": row["Temperature"],
                }
            )
    full_index_df = pd.DataFrame(full_index_list)

    # 5. Merge observed stacks onto the full grid
    complete_stack_data = pd.merge(
        full_index_df,
        combined_stacks[["GameID", "RoundID", "PlayerID", "StackBefore"]],
        on=["GameID", "RoundID", "PlayerID"],
        how="left",
    )

    # 6. Forward-fill stacks within each game/player group
    complete_stack_data = complete_stack_data.sort_values(
        by=["GameID", "PlayerID", "RoundID"]
    )
    complete_stack_data["StackBefore"] = complete_stack_data.groupby(
        ["GameID", "PlayerID"]
    )["StackBefore"].ffill()

    # Handle potential remaining NaNs if a player *never* had an action logged (should be rare if Round 0 worked)
    complete_stack_data["StackBefore"].fillna(
        starting_stack, inplace=True
    )  # Fill with starting stack as last resort

    print(
        f"Generated complete stack data for {len(complete_stack_data)} player-round entries."
    )
    return complete_stack_data


# ==============================================================================
# Plotting Function (Chip Stack Evolution Subplots - REVISED)
# ==============================================================================
def plot_chip_stacks(
    stack_data, fig_title, filename=None, starting_stack=STARTING_STACK
):
    """
    Plots chip stack progression in subplots based on temperature.
    Each game is a separate line, colored by the base model name.
    """
    if stack_data.empty:
        print(f"Skipping plot '{fig_title}' - DataFrame is empty.")
        return
    required_cols = [
        "ModelShortName",
        "Temperature",
        "GameID",
        "RoundID",
        "StackBefore",
    ]  # Removed model_temp_id, PlayerID not strictly needed here
    if not all(col in stack_data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in stack_data.columns]
        print(
            f"Skipping plot '{fig_title}' - Stack data missing required columns: {missing}"
        )
        return

    # --- Data Preparation ---
    temps = sorted(stack_data["Temperature"].unique())
    models = sorted(stack_data["ModelShortName"].unique())
    n_temps = len(temps)
    n_models = len(models)
    if n_temps == 0 or n_models == 0:
        print(
            f"Skipping plot '{fig_title}' - Not enough data (models={n_models}, temps={n_temps})."
        )
        return

    # Determine subplot layout
    ncols = n_temps if n_temps <= 3 else 3  # Max 3 columns
    nrows = math.ceil(n_temps / ncols)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 6, nrows * 5.5 + 0.5), sharey=True, squeeze=False
    )  # Added height for title
    axs = axs.flatten()

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except IOError:
        try:
            plt.style.use("seaborn-whitegrid")
        except IOError:
            print("Warning: Seaborn styles not found. Using default style.")

    print(
        f"Plotting chip stacks for {n_models} models across {n_temps} temperatures..."
    )
    legend_handles = []
    plotted_models = set()

    # --- Plot Lines ---
    for t_idx, temp in enumerate(temps):
        if t_idx >= len(axs):
            break  # Safety break
        ax = axs[t_idx]
        temp_data = stack_data[stack_data["Temperature"] == temp]
        print(f"  - Plotting Temp: {temp} on subplot {t_idx}")

        ax.set_title(f"Temperature: {temp}", fontsize=14)
        if temp_data.empty:
            ax.text(
                0.5,
                0.5,
                "No data for this temperature",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        for model_name in models:
            model_temp_data = temp_data[temp_data["ModelShortName"] == model_name]
            if model_temp_data.empty:
                continue

            color = MODEL_COLORS.get(model_name, DEFAULT_MODEL_COLOR)
            num_games = model_temp_data["GameID"].nunique()
            # print(f"    - Plotting Model: {model_name} ({num_games} games)") # Verbose

            # Plot each game line for this model at this temp
            for game_id, group_game in model_temp_data.groupby("GameID"):
                group_game = group_game.sort_values("RoundID")
                ax.plot(
                    group_game["RoundID"],
                    group_game["StackBefore"],
                    color=color,
                    alpha=0.4,
                    linewidth=1.2,
                    label="_nolegend_",
                )

            # Add handle for the legend only once per model
            if model_name not in plotted_models:
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=2, label=model_name)
                )
                plotted_models.add(model_name)

        # --- Subplot Formatting ---
        ax.axhline(starting_stack, color="dimgrey", linestyle="--", linewidth=1.5)
        if t_idx % ncols == 0:
            ax.set_ylabel("Chip Stack", fontsize=13)
        ax.set_xlabel("Round Number", fontsize=13)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0
        ax.margins(x=0.02)

    # Hide unused subplots
    for k in range(t_idx + 1, len(axs)):
        fig.delaxes(axs[k])

    # --- Overall Figure Formatting ---
    fig.suptitle(
        fig_title, fontsize=18, fontweight="bold", y=0.99
    )  # Adjust title position

    # Add starting stack line to legend handles
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="dimgrey",
            linestyle="--",
            lw=1.5,
            label=f"Start ({starting_stack})",
        )
    )

    # Create the legend for the whole figure
    fig.legend(
        handles=legend_handles,
        title="Model",
        loc="upper right",
        bbox_to_anchor=(1.08 if ncols > 1 else 1.15, 0.9),
        fontsize=10,
        title_fontsize=11,
    )  # Adjust anchor/location

    fig.tight_layout(rect=[0, 0, 0.9, 0.96])  # Adjust layout rect for title and legend

    # Save or show
    if filename:
        dir_name = os.path.dirname(filename)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
        plt.savefig(filename, bbox_inches="tight", dpi=150)
        print(f"Plot saved to: {filename}")
        plt.close(fig)
    else:
        plt.show()


# ==============================================================================
# Main Execution Block (Updated to use bar plots with CI)
# ==============================================================================
def main():
    print(f"Starting Poker Log Analysis V4...")
    if not os.path.exists(OUTPUT_PLOT_DIR):
        os.makedirs(OUTPUT_PLOT_DIR)
        print(f"Created base plot directory: {OUTPUT_PLOT_DIR}")
    try:
        logs = load_all_logs(base_pattern="simulation_*")
        if logs.empty:
            print("Error: No data loaded. Exiting.")
            return
        if "model_temp_id" not in logs.columns:
            print("Error: 'model_temp_id' column not created.")
            return

        print("\nComputing Metrics with Confidence Intervals...")
        preflop_metrics, preflop_ci = compute_preflop_metrics_with_ci(logs)
        postflop_metrics, postflop_ci = compute_postflop_metrics_with_ci(logs)
        bet_sizing_metrics, bet_sizing_ci = compute_bet_sizing_with_ci(logs, big_blind=BIG_BLIND)
        positional_metrics, positional_ci = compute_positional_metrics_with_ci(logs)
        print("Metrics computation complete.")

        # --- Plot Metric Trends as Bar Charts ---
        print("\n--- Plotting Pre-flop Metric Bar Charts ---")
        print(preflop_metrics.to_string())
        try:
            plot_metric_bars_vs_temp(
                preflop_metrics,
                ["VPIP", "PFR"],
                "Pre-flop Metrics vs. Temperature",
                os.path.join(OUTPUT_PLOT_DIR, "preflop_trends.png"),
                sharey=True,
                ci_data=preflop_ci
            )
        except Exception as e:
            print(f"Could not generate pre-flop bar chart: {e}")
            traceback.print_exc()

        print("\n--- Plotting Post-flop Metric Bar Charts ---")
        print(postflop_metrics.to_string())
        try:
            plot_metric_bars_vs_temp(
                postflop_metrics,
                ["AFq", "WTSD", "WSD"],
                "Post-flop Metrics vs. Temperature",
                os.path.join(OUTPUT_PLOT_DIR, "postflop_trends.png"),
                sharey=False,
                ci_data=postflop_ci
            )
        except Exception as e:
            print(f"Could not generate post-flop bar chart: {e}")
            traceback.print_exc()

        print("\n--- Plotting Bet Sizing Metric Bar Charts ---")
        print(bet_sizing_metrics.to_string())
        try:
            plot_metric_bars_vs_temp(
                bet_sizing_metrics,
                ["PF_Raise_xBB", "Avg_Pct_Pot_Raise"],
                "Bet Sizing Metrics vs. Temperature",
                os.path.join(OUTPUT_PLOT_DIR, "bet_sizing_trends.png"),
                sharey=False,
                ci_data=bet_sizing_ci
            )
        except Exception as e:
            print(f"Could not generate bet sizing bar chart: {e}")
            traceback.print_exc()

        print("\n--- Plotting Positional Metric Bar Charts ---")
        print(positional_metrics.to_string())
        key_positions = ["BTN", "SB", "BB", "BTN_HU"]
        for pos in key_positions:
            pos_vpip_col, pos_pfr_col = f"vpip_{pos}", f"pfr_{pos}"
            current_metric_cols = [
                col
                for col in [pos_vpip_col, pos_pfr_col]
                if col in positional_metrics.columns
            ]
            if current_metric_cols:
                print(f"\nPlotting Position: {pos}")
                try:
                    # Extract CI data only relevant to this position
                    pos_ci_data = {k: v for k, v in positional_ci.items() 
                                  if any(pos in k[2] for k in [k])}
                    
                    plot_metric_bars_vs_temp(
                        positional_metrics,
                        current_metric_cols,
                        f"Positional Metrics ({pos}) vs. Temperature",
                        os.path.join(OUTPUT_PLOT_DIR, f"positional_{pos}_trends.png"),
                        sharey=True,
                        ci_data=pos_ci_data
                    )
                except Exception as e:
                    print(f"Could not generate positional bar chart for {pos}: {e}")
                    traceback.print_exc()
            else:
                print(f"Skipping bar chart for position {pos} - No data columns found.")

        # --- Extract and Plot Chip Stacks (Unchanged - Keeps Line Plot) ---
        print("\n--- Plotting Chip Stack Evolutions ---")
        try:
            stack_data = extract_round_start_stacks(logs, starting_stack=STARTING_STACK)
            plot_chip_stacks(
                stack_data,
                "Chip Stack Evolution per Game (Subplots by Temp, Color by Model)",
                os.path.join(OUTPUT_PLOT_DIR, "chip_stack_evolution.png"),
                starting_stack=STARTING_STACK,
            )
        except Exception as e:
            print(f"Could not generate chip stack evolution plot: {e}")
            traceback.print_exc()

        print(f"\nAnalysis complete. Plots saved in '{OUTPUT_PLOT_DIR}'")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
