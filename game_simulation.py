# -*- coding: utf-8 -*-
"""
Simulates multiple games of Texas Hold'em poker between AI models and stores results in CSV files.
Uses the 'pokerlib' library for game logic and the Together AI API for AI opponents.
Integrates 'deuces' library for hand evaluation (though not used for probabilities in this version).

Includes manual winner detection for premature round ends and fixes board card logging.
Handles temperature logging correctly.
"""

import csv
import logging
import os
import random
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import requests
from dotenv import load_dotenv
# Necessary imports from pokerlib
from pokerlib import Player, PlayerSeats, Table
from pokerlib.enums import (Hand, Rank, RoundPrivateOutId, RoundPublicInId,
                            RoundPublicOutId, Suit, TablePrivateOutId,
                            TablePublicInId, TablePublicOutId, Turn)

# --- Deuces Integration (for hand evaluation only) ---
try:
    from deuces import Card, Evaluator
except ImportError:
    print("Error: 'deuces' library not found. Please install it: pip install deuces")
    sys.exit(1)
# --------------------------

# ==============================================================================
# Constants and Configuration
# ==============================================================================

# --- Simulation Settings ---
GAMES = 20  # Number of games to simulate.
ROUNDS = 20  # Number of rounds (hands) per game.

# --- General Settings ---
VERBOSE = False  # If True, print extra AI debugging information.
LOG_FILE_NAME = "log_simulation.txt"  # File to log AI interactions and game events.

# --- Game Parameters ---
BUYIN = 1000 # Default starting money for each player.
SMALL_BLIND = 5  # Small blind amount.
BIG_BLIND = 10  # Big blind amount.

# --- AI Configuration ---
# Ensure at least 2 models for simulation
AI_MODEL_LIST = [  # Models for AI opponents (requires Together AI API Key).
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
]
# Dictionary mapping full model names to desired short names for logging/internal use.
AI_MODEL_SHORT_NAMES = {
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "Qwen 2.5 7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral v0.3 7B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "Llama 3.1 8B",
    "upstage/SOLAR-10.7B-Instruct-v1.0": "Upstage Solar 11B",
    "mistralai/Mistral-Small-24B-Instruct-2501": "Mistral Small 3 24B",
    "google/gemma-2-27b-it": "Gemma 2 27B",
    "Qwen/QwQ-32B-Preview": "Qwen QwQ 32B",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama 3.3 70B",
    "Qwen/Qwen2.5-72B-Instruct-Turbo": "Qwen 2.5 72B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
    "microsoft/WizardLM-2-8x22B": "WizardLM-2 8x22B",
    "databricks/dbrx-instruct": "DBRX Instruct",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3",
    "deepseek-ai/DeepSeek-R1": "DeepSeek R1",
}

# AI_TEMPERATURE = 0.5  # Sampling temperature for AI model responses.
AI_REQUEST_TIMEOUT = 90  # Timeout in seconds for API requests to Together AI (increased for potentially slower models).
AI_RETRY_DELAY = 2  # Delay in seconds before retrying a failed API request.


# --- Derived Constants & Initial Checks ---
NUM_PLAYERS = len(AI_MODEL_LIST)
AI_ONLY_MODE = True  # This script is AI only

if NUM_PLAYERS < 2:
    print(f"Error: Need at least 2 AI models in AI_MODEL_LIST (found {NUM_PLAYERS}).")
    sys.exit(1)


# --- ANSI Color Codes (for logging/debug output if needed) ---
class Colors:
    RESET = ""
    BOLD = ""
    UNDERLINE = ""
    BLACK = ""
    RED = ""
    GREEN = ""
    YELLOW = ""
    BLUE = ""
    MAGENTA = ""
    CYAN = ""
    WHITE = ""
    BRIGHT_BLACK = ""
    BRIGHT_RED = ""
    BRIGHT_GREEN = ""
    BRIGHT_YELLOW = ""
    BRIGHT_BLUE = ""
    BRIGHT_MAGENTA = ""
    BRIGHT_CYAN = ""
    BRIGHT_WHITE = ""
    # Simple implementation to avoid errors if colors are used accidentally
    # Could be re-enabled for console debugging if needed


# --- Card Formatting Maps ---
RANK_MAP_POKERLIB = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}
SUIT_MAP_POKERLIB = {
    Suit.SPADE: "s",
    Suit.CLUB: "c",
    Suit.DIAMOND: "d",
    Suit.HEART: "h",
}

# ==============================================================================
# Logging Setup
# ==============================================================================
if os.path.exists(LOG_FILE_NAME):
    os.remove(LOG_FILE_NAME)

ai_logger = logging.getLogger("AIPokerSimLogDetailed")
ai_logger.setLevel(logging.INFO)
ai_logger.propagate = False

# Ensure log file directory exists
log_dir = Path(LOG_FILE_NAME).parent
if log_dir:
    log_dir.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_formatter = logging.Formatter(
    "%(asctime)s - G%(game_num)d R%(round_num)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(file_formatter)
ai_logger.addHandler(file_handler)


# Add game/round info to logs using LogRecord attributes
# Need a filter to add these attributes if they aren't present
class ContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "game_num"):
            record.game_num = 0
        if not hasattr(record, "round_num"):
            record.round_num = 0
        return True


ai_logger.addFilter(ContextFilter())


def log_info(message, game_num=0, round_num=0):
    ai_logger.info(message, extra={"game_num": game_num, "round_num": round_num})


def log_warning(message, game_num=0, round_num=0):
    ai_logger.warning(message, extra={"game_num": game_num, "round_num": round_num})


def log_error(message, game_num=0, round_num=0, exc_info=False):
    ai_logger.error(
        message, extra={"game_num": game_num, "round_num": round_num}, exc_info=exc_info
    )


log_info("--- AI Poker Simulation Log Initialized (Detailed) ---")

# ==============================================================================
# API Key Loading
# ==============================================================================
load_dotenv()
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")

if not TOGETHER_AI_API_KEY:
    error_msg = "ERROR: TOGETHER_AI_API_KEY environment variable is missing or empty."
    print(error_msg)  # Print to console as well
    log_error(error_msg)
    sys.exit(1)
else:
    log_info("TOGETHER_AI_API_KEY loaded successfully.")

# ==============================================================================
# Helper Functions
# ==============================================================================


def format_card_for_ai(card):
    """Formats a card tuple into a simple string for AI prompts (e.g., "As", "Kh")."""
    fallback_card = "??"
    if not card or not hasattr(card, "__len__") or len(card) != 2:
        return fallback_card
    try:
        rank, suit = card
        rank_str = RANK_MAP_POKERLIB.get(rank)
        suit_str = SUIT_MAP_POKERLIB.get(suit)
        if rank_str is None or suit_str is None:
            return fallback_card
        return f"{rank_str}{suit_str}"
    except (TypeError, ValueError, IndexError):
        return fallback_card


def format_cards_for_ai(cards):
    """Formats an iterable of cards into a space-separated string for AI prompts."""
    if cards and hasattr(cards, "__iter__"):
        valid_cards = [c for c in cards if isinstance(c, (list, tuple)) and len(c) == 2]
        return " ".join(format_card_for_ai(c) for c in valid_cards)
    return ""


def format_hand_enum(hand_enum):
    """Converts a pokerlib Hand enum member into a readable title-case string."""
    return hand_enum.name.replace("_", " ").title() if hand_enum else "Unknown Hand"


def get_player_position(player_index, num_players, button_index):
    """Determines player position category (SB, BB, BTN, CO, HJ, MP, EP etc.)."""
    if num_players < 2:
        return "N/A"
    if num_players == 2:
        if player_index == button_index:
            return "BTN/SB"
        else:
            return "BB"

    relative_pos = (player_index - button_index + num_players) % num_players

    if relative_pos == 0:
        return "BTN"
    if relative_pos == 1:
        return "SB"
    if relative_pos == 2:
        return "BB"

    pos_from_button = num_players - 1 - relative_pos

    if num_players <= 6:
        if pos_from_button == 0:
            return "CO"
        if pos_from_button == 1:
            return "HJ"
        if pos_from_button == 2:
            return "UTG"
        if pos_from_button == 3 and num_players == 6:
            return "UTG"
        return f"POS{relative_pos}"
    else:
        if pos_from_button == 0:
            return "CO"
        if pos_from_button == 1:
            return "HJ"
        mp_count = (num_players - 6) // 2
        if pos_from_button >= 2 and pos_from_button < 2 + mp_count:
            return f"MP{pos_from_button - 1}"
        utg_count = num_players - 3 - mp_count - 2
        if pos_from_button >= 2 + mp_count:
            return f"UTG{pos_from_button - mp_count - 1}"
        return f"POS{relative_pos}"


# ==============================================================================
# AI Interaction Functions
# ==============================================================================


def query_together_ai(model_name, messages, temperature, game_num=0, round_num=0):
    """Sends a prompt to the Together AI API and returns the model's response."""
    api_endpoint = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "PokerSim-AI-Detailed-v3",  # Updated User-Agent
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 150,
    }
    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        try:
            response = requests.post(
                api_endpoint, headers=headers, json=payload, timeout=AI_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            if (
                data
                and "choices" in data
                and len(data["choices"]) > 0
                and "message" in data["choices"][0]
                and "content" in data["choices"][0]["message"]
            ):
                return data["choices"][0]["message"]["content"].strip()
            else:
                warn_msg = (
                    f"W: Unexpected API response structure from {model_name}: {data}"
                )
                log_warning(warn_msg, game_num, round_num)
                return None

        except requests.exceptions.Timeout:
            warn_msg = f"W: API request to {model_name} timed out (attempt {attempt + 1}/{max_retries})."
            log_warning(warn_msg, game_num, round_num)
            attempt += 1
            if attempt >= max_retries:
                return None
            time.sleep(AI_RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            err_msg = f"Error querying {model_name}: {e}"
            if hasattr(e, "response") and e.response is not None:
                err_msg += (
                    f" | Status: {e.response.status_code} | Body: {e.response.text}"
                )
            log_error(err_msg, game_num, round_num)
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == 429
            ):
                attempt += 1
                retry_delay = AI_RETRY_DELAY * (attempt + 1)
                log_warning(
                    f"Rate limit hit. Retrying in {retry_delay}s (attempt {attempt}/{max_retries})...",
                    game_num,
                    round_num,
                )
                time.sleep(retry_delay)
            else:
                return None

        except Exception as e:
            err_msg = f"Unexpected error during API call to {model_name}: {e}"
            log_error(err_msg, game_num, round_num, exc_info=True)
            return None

    fail_msg = f"API call to {model_name} failed after {max_retries} retries."
    log_error(fail_msg, game_num, round_num)
    return None


def parse_ai_action(response_text, game_num=0, round_num=0):
    """Parses the AI's text response to determine the poker action and raise amount."""
    if not response_text:
        log_warning(
            "AI response was empty, defaulting to CHECK/FOLD.", game_num, round_num
        )
        return None, {}

    response_lines = response_text.lower().split("\n")
    action = None
    raise_by = 0
    action_line_found = False

    for line in response_lines:
        line = line.strip()
        if not line:
            continue

        action_word = ""
        potential_amount = ""

        if line.startswith("action:"):
            parts = line.split(":", 1)[1].strip().split()
            if parts:
                action_word = parts[0]
                if action_word == "raise" and "amount:" in line:
                    try:
                        amount_str = line.split("amount:", 1)[1].strip()
                        if amount_str.isdigit():
                            potential_amount = amount_str
                    except Exception:
                        pass
        elif line.startswith("fold"):
            action_word = "fold"
        elif line.startswith("check"):
            action_word = "check"
        elif line.startswith("call"):
            action_word = "call"
        elif line.startswith("raise"):
            action_word = "raise"
            parts = line.split()
            found_num = False
            for i, part in enumerate(parts):
                num_part = part.strip(":").strip()
                if num_part.isdigit():
                    potential_amount = num_part
                    found_num = True
                    if i > 0 and parts[i - 1] in ["raise", "by", "amount", "amount:"]:
                        break
            if not found_num:
                for part in parts:
                    if part.isdigit():
                        potential_amount = part
                        break

        if action_word == "fold":
            action = RoundPublicInId.FOLD
            action_line_found = True
            break
        if action_word == "check":
            action = RoundPublicInId.CHECK
            action_line_found = True
            break
        if action_word == "call":
            action = RoundPublicInId.CALL
            action_line_found = True
            break
        if action_word == "raise":
            action = RoundPublicInId.RAISE
            action_line_found = True
            if potential_amount and potential_amount.isdigit():
                raise_by = int(potential_amount)
            else:
                log_warning(
                    f"AI response indicated 'raise' but no valid amount found. Raw: '{response_text}'",
                    game_num,
                    round_num,
                )
                action = None
            break

    if action == RoundPublicInId.RAISE and raise_by <= 0:
        log_warning(
            f"AI responded RAISE but failed to specify valid amount > 0 ({raise_by}). Raw: '{response_text}'. Applying fallback logic.",
            game_num,
            round_num,
        )
        action = None
        raise_by = 0
    elif not action_line_found:
        log_warning(
            f"Could not parse valid action keyword from AI response: '{response_text}'. Applying fallback logic.",
            game_num,
            round_num,
        )
        action = None
        raise_by = 0

    kwargs = {"raise_by": raise_by} if action == RoundPublicInId.RAISE else {}
    return action, kwargs


def format_state_for_ai(table_state, player_id_acting, game_num=0, round_num=0):
    """Formats the current game state into a markdown string suitable for the AI prompt."""
    lines = []
    if not table_state.round or not hasattr(table_state.round, "turn"):
        log_error("Game state missing round/turn info.", game_num, round_num)
        return "Error: Missing round/turn info."
    if not hasattr(table_state, "_player_cards"):
        log_error("Table state missing _player_cards info.", game_num, round_num)
        return "Error: Missing player card info."

    try:
        lines.append(
            f"## Poker Hand State - Game {game_num}, Round {round_num} (Hand ID {table_state.current_hand_id})"
        )
        street_name = table_state.current_street
        lines.append(f"**Current Stage:** {street_name}")

        board_cards_tuples = table_state._get_current_board_tuples()
        board_str = format_cards_for_ai(board_cards_tuples)
        lines.append(f"**Board:** [ {board_str} ] ({len(board_cards_tuples)} cards)")

        pot_total = table_state.pot_at_action_start
        lines.append(f"**Total Pot (before your action):** ${pot_total}")

        lines.append("\n**Players:** (Order is position relative to dealer)")
        acting_player_obj = None
        players_in_pokerlib_round = getattr(table_state.round, "players", [])
        player_details = []

        for p in players_in_pokerlib_round:
            if not p or not hasattr(p, "id"):
                continue

            player_id = p.id
            is_acting = player_id == player_id_acting
            if is_acting:
                acting_player_obj = p

            name_str = getattr(p, "name", f"P{player_id}")
            money_val = getattr(p, "money", 0)
            cards_str = "( ? ? )"
            if player_id in table_state._player_cards:
                if is_acting:
                    cards_str = f"( {format_cards_for_ai(table_state._player_cards[player_id])} )"

            status = []
            if getattr(p, "is_folded", False):
                status.append("FOLDED")
            if getattr(p, "is_all_in", False):
                status.append("ALL-IN")

            position_str = table_state.player_positions.get(player_id, "UnknownPos")

            if position_str == "BTN":
                status.append("D")
            if position_str == "SB":
                status.append("SB")
            if position_str == "BB":
                status.append("BB")
            status_str = f"[{' '.join(status)}]" if status else ""

            current_bet = 0
            turn_stake_list = getattr(p, "turn_stake", [])
            current_turn_value = getattr(table_state.round.turn, "value", -1)
            if (
                isinstance(turn_stake_list, list)
                and current_turn_value >= 0
                and len(turn_stake_list) > current_turn_value
            ):
                current_bet = turn_stake_list[current_turn_value]
            bet_str = f"Bet:${current_bet}"

            player_details.append(
                {
                    "id": player_id,
                    "name": name_str,
                    "money": money_val,
                    "cards": cards_str,
                    "bet": bet_str,
                    "status": status_str,
                    "acting": is_acting,
                    "position": position_str,
                }
            )

        max_name_len = (
            max(len(pd["name"]) for pd in player_details) if player_details else 10
        )
        for pd in player_details:
            prefix = ">>> **YOU**" if pd["acting"] else "   -"
            name_formatted = pd["name"].ljust(max_name_len)
            money_formatted = f"Stack:${pd['money']}".ljust(12)
            cards_formatted = pd["cards"].ljust(10)
            bet_formatted = pd["bet"].ljust(10)
            pos_formatted = pd["position"].ljust(8)
            lines.append(
                f"{prefix} {pos_formatted} {name_formatted} {money_formatted} {bet_formatted} {cards_formatted} {pd['status']}"
            )

        lines.append("\n**Action Context:**")
        current_round_bet_level = getattr(table_state.round, "turn_stake", 0)
        lines.append(f"- Current Bet Level to Match: ${current_round_bet_level}")

        amount_to_call = table_state.to_call
        lines.append(f"- Amount You Need to Call: ${amount_to_call}")
        acting_player_money = getattr(acting_player_obj, "money", 0)
        lines.append(f"- Your Stack: ${acting_player_money}")
        min_raise_by = table_state.min_raise

        possible_actions_list = []
        if table_state.can_fold:
            possible_actions_list.append("FOLD")
        if table_state.can_check:
            possible_actions_list.append("CHECK")
        elif amount_to_call >= 0 and acting_player_money > 0:
            call_amount_str = min(amount_to_call, acting_player_money)
            possible_actions_list.append(f"CALL({call_amount_str})")
        if table_state.can_raise:
            max_raise_by = acting_player_money - amount_to_call
            actual_min_raise = (
                min(min_raise_by, max_raise_by) if max_raise_by > 0 else 0
            )
            if actual_min_raise > 0:
                possible_actions_list.append(
                    f"RAISE(min:{actual_min_raise} max:{max_raise_by})"
                )

        lines.append(f"- Possible Actions: {', '.join(possible_actions_list)}")

        lines.append(
            f"\n**Task:** You are player {acting_player_obj.name} ({acting_player_obj.id}). Decide your action."
        )
        lines.append(f"Respond ONLY with the action name: FOLD, CHECK, CALL, or RAISE.")
        lines.append(
            f"If RAISING, add ' AMOUNT: X' on the same line, where X is the integer amount to raise BY (additional chips on top of the call amount). Example: 'RAISE AMOUNT: {min(max(min_raise_by, 2 * BIG_BLIND), max(0, acting_player_money - amount_to_call))}'"
        )

        return "\n".join(lines)

    except Exception as e:
        log_error(
            f"Error during format_state_for_ai: {e}", game_num, round_num, exc_info=True
        )
        return f"Error: Could not format game state - {e}"


# ==============================================================================
# Custom Table Class for Data Collection
# ==============================================================================


class SimulationTable(Table):
    """
    Custom Table class for AI simulation and detailed CSV data collection.
    Includes manual premature winner detection and correct temperature logging.
    """

    CSV_HEADERS = [
        "GameID",
        "RoundID",
        "ActionID",
        "Street",
        "PlayerID",
        "PlayerName",
        "ModelName",
        "Temperature",
        "Position",
        "HoleCard1",
        "HoleCard2",
        "BoardCards",
        "StackBefore",
        "PotTotalBefore",
        "BetToCall",
        "ActionType",
        "ActionAmount",
        "AmountWon",
        "FinalHandRank",
    ]

    ACTION_DEALT_HOLE = "DEALT_HOLE"
    ACTION_POST_SB = "POST_SB"
    ACTION_POST_BB = "POST_BB"
    ACTION_FOLD = "FOLD"
    ACTION_CHECK = "CHECK"
    ACTION_CALL = "CALL"
    ACTION_BET = "BET"
    ACTION_RAISE = "RAISE"
    ACTION_ALL_IN = "ALL_IN"
    ACTION_DEALT_FLOP = "DEALT_FLOP"
    ACTION_DEALT_TURN = "DEALT_TURN"
    ACTION_DEALT_RIVER = "DEALT_RIVER"
    ACTION_SHOWS_HAND = "SHOWS_HAND"
    ACTION_WINS_POT = "WINS_POT"

    def __init__(self, game_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_num = game_num
        self.round_num = 0

        self.game_data = []
        self.current_hand_id = 0
        self.current_hand_actions = []
        self.action_counter = 0
        self.current_street = "Preflop"
        self.player_positions = {}
        self.pot_at_action_start = 0
        self._player_cards = {}
        self.board_cards = []  # Correctly store board cards as list of tuples

        self._current_action_player_id = None
        self.round_over_flag = False
        self.to_call = 0
        self.can_check = False
        self.can_raise = False
        self.can_fold = True
        self.min_raise = 0
        self.street_has_bet = False
        self.winner_declared_this_round = False
        self._last_action_temperature = (
            None  # <<< Added to store temperature temporarily
        )

        self.ai_message_history = defaultdict(lambda: defaultdict(list))
        self.ai_model_assignments = {}

        self.evaluator = Evaluator()

    def assign_ai_models(self):
        ai_player_ids = sorted(
            [p.id for p in self.seats.getPlayerGroup() if p and hasattr(p, "id")]
        )
        num_ai = len(ai_player_ids)
        if num_ai == 0 or not AI_MODEL_LIST:
            log_info(
                "No AI players or models defined, skipping assignment.",
                self.game_num,
                self.round_num,
            )
            return

        log_msg_assign = ["AI Model Assignments:"]
        for i, pid in enumerate(ai_player_ids):
            if pid not in self.ai_model_assignments:
                model_full_name = AI_MODEL_LIST[i % len(AI_MODEL_LIST)]
                self.ai_model_assignments[pid] = model_full_name
                model_short_name = AI_MODEL_SHORT_NAMES.get(
                    model_full_name, model_full_name.split("/")[-1]
                )
                player_name = self._get_player_name(pid)
                log_msg_assign.append(
                    f"  - Player {pid} ({player_name}): {model_short_name} ({model_full_name})"
                )
        log_info("\n".join(log_msg_assign), self.game_num, self.round_num)

    def _get_player_name(self, player_id):
        player = self.seats.getPlayerById(player_id)
        return (
            getattr(player, "name", f"P{player_id}")
            if player
            else f"P{player_id}_NotFound"
        )

    def _get_current_board_tuples(self):
        return self.board_cards

    def _get_current_board_str(self):
        return format_cards_for_ai(self.board_cards)

    def _newRound(self, round_id):
        self.round_num = round_id
        self.current_hand_id = self.game_num * 1000 + self.round_num
        log_info(
            f"Starting Hand ID {self.current_hand_id}", self.game_num, self.round_num
        )

        current_players = self.seats.getPlayerGroup()
        if not current_players or len(current_players) < 2:
            log_error(
                f"Cannot start round {self.round_num} - insufficient players ({len(current_players)}).",
                self.game_num,
                self.round_num,
            )
            self.round = None
            self.round_over_flag = True
            return

        for player in current_players:
            if hasattr(player, "resetState"):
                try:
                    player.resetState()
                except Exception as e:
                    log_warning(
                        f"Could not reset state for P{player.id}: {e}",
                        self.game_num,
                        self.round_num,
                    )

        self._player_cards.clear()
        self.board_cards = []
        self.current_hand_actions = []
        self.action_counter = 1
        self.current_street = "Preflop"
        self.round_over_flag = False
        self._current_action_player_id = None
        self.to_call = 0
        self.can_check = False
        self.can_raise = False
        self.can_fold = True
        self.min_raise = self.big_blind
        self.pot_at_action_start = 0
        self.street_has_bet = False
        self.winner_declared_this_round = False
        self._last_action_temperature = None  # <<< Reset temp storage

        try:
            self.round = self.RoundClass(
                self.round_num,
                current_players,
                self.button,
                self.small_blind,
                self.big_blind,
            )
            log_info(
                f"Successfully initialized pokerlib Round {self.round_num}.",
                self.game_num,
                self.round_num,
            )

            self.player_positions.clear()
            players_in_pokerlib_round = getattr(self.round, "players", [])
            button_idx = getattr(self.round, "button", -1)
            num_players_round = len(players_in_pokerlib_round)
            if button_idx != -1 and num_players_round > 0:
                for idx, p in enumerate(players_in_pokerlib_round):
                    if p and hasattr(p, "id"):
                        self.player_positions[p.id] = get_player_position(
                            idx, num_players_round, button_idx
                        )
                log_info(
                    f"Player positions set: {self.player_positions}",
                    self.game_num,
                    self.round_num,
                )
            else:
                log_warning(
                    "Could not determine button or players to set positions.",
                    self.game_num,
                    self.round_num,
                )

        except Exception as e:
            log_error(
                f"Failed to initialize pokerlib Round {self.round_num}: {e}",
                self.game_num,
                self.round_num,
                exc_info=True,
            )
            self.round = None
            self.round_over_flag = True

    def _record_action(
        self,
        action_type,
        player_id=None,
        action_amount=0,
        amount_won=0,
        final_hand_rank=None,
        temperature=None,
    ):
        # This function now correctly accepts and uses the temperature argument
        if not self.round and action_type not in [self.ACTION_WINS_POT]:
            log_warning(
                f"Cannot record action '{action_type}' - Round object is missing.",
                self.game_num,
                self.round_num,
            )
            return

        player = self.seats.getPlayerById(player_id) if player_id else None
        player_name = getattr(player, "name", None) if player else None
        model_name = self.ai_model_assignments.get(player_id) if player_id else None
        position = self.player_positions.get(player_id) if player_id else None
        hole_cards = self._player_cards.get(player_id) if player_id else None
        hole_card1_str = (
            format_card_for_ai(hole_cards[0])
            if hole_cards and len(hole_cards) > 0
            else None
        )
        hole_card2_str = (
            format_card_for_ai(hole_cards[1])
            if hole_cards and len(hole_cards) > 1
            else None
        )

        is_player_decision = action_type in [
            self.ACTION_FOLD,
            self.ACTION_CHECK,
            self.ACTION_CALL,
            self.ACTION_BET,
            self.ACTION_RAISE,
        ]
        pot_before = self.pot_at_action_start
        stack_before = getattr(player, "money", 0) if player else 0
        bet_to_call = self.to_call if player_id == self._current_action_player_id else 0

        if not is_player_decision:
            # Use current pot for non-decision events like blinds/deals
            current_pot_list = (
                getattr(self.round, "pot_size", [0]) if self.round else [0]
            )
            pot_before = (
                sum(current_pot_list)
                if isinstance(current_pot_list, list)
                else current_pot_list
            )
            bet_to_call = 0

        board_str = self._get_current_board_str()

        action_data = {
            "GameID": self.game_num,
            "RoundID": self.round_num,
            "ActionID": self.action_counter,
            "Street": self.current_street,
            "PlayerID": player_id,
            "PlayerName": player_name,
            "ModelName": model_name,
            "Temperature": (
                temperature if is_player_decision else None
            ),  # Log temp only for AI decisions
            "Position": position,
            "HoleCard1": hole_card1_str,
            "HoleCard2": hole_card2_str,
            "BoardCards": board_str,
            "StackBefore": stack_before,
            "PotTotalBefore": pot_before,
            "BetToCall": bet_to_call,
            "ActionType": action_type,
            "ActionAmount": action_amount,
            "AmountWon": amount_won,
            "FinalHandRank": final_hand_rank,
        }

        self.current_hand_actions.append(action_data)
        self.action_counter += 1

    def _finalize_hand_data(self):
        log_info(
            f"Finalizing data for Hand ID {self.current_hand_id}. Actions recorded: {len(self.current_hand_actions)}",
            self.game_num,
            self.round_num,
        )
        self.game_data.extend(self.current_hand_actions)
        log_info(
            f"Added {len(self.current_hand_actions)} actions from hand {self.current_hand_id} to game data. Total game actions: {len(self.game_data)}",
            self.game_num,
            self.round_num,
        )
        self.current_hand_actions = []

    def publicOut(self, out_id, **kwargs):
        player_id = kwargs.get("player_id")
        player = self.seats.getPlayerById(player_id) if player_id else None
        log_extra = {"game_num": self.game_num, "round_num": self.round_num}

        is_round_event = isinstance(out_id, RoundPublicOutId)

        if (
            is_round_event
            and not self.round
            and not self.round_over_flag
            and out_id != RoundPublicOutId.ROUNDFINISHED
        ):
            log_warning(
                f"Round object is None but round_over_flag is False before event {out_id}. Forcing round end.",
                **log_extra,
            )
            self.round_over_flag = True
            self._finalize_hand_data()
            return

        if (is_round_event and self.round) or out_id == RoundPublicOutId.ROUNDFINISHED:
            if out_id == RoundPublicOutId.NEWTURN:
                turn_enum = kwargs.get("turn")
                log_info(
                    f"DEBUG: Handling NEWTURN event for Turn: {turn_enum}. Kwargs: {kwargs}",
                    **log_extra,
                )
                action_type = None
                if turn_enum == Turn.FLOP:
                    self.current_street = "Flop"
                    action_type = self.ACTION_DEALT_FLOP
                    self.street_has_bet = False
                elif turn_enum == Turn.TURN:
                    self.current_street = "Turn"
                    action_type = self.ACTION_DEALT_TURN
                    self.street_has_bet = False
                elif turn_enum == Turn.RIVER:
                    self.current_street = "River"
                    action_type = self.ACTION_DEALT_RIVER
                    self.street_has_bet = False

                if self.round and hasattr(self.round, "board"):
                    try:
                        board_from_lib = getattr(self.round, "board", [])
                        self.board_cards = [
                            tuple(c)
                            for c in board_from_lib
                            if isinstance(c, (list, tuple)) and len(c) == 2
                        ]
                        log_info(
                            f"DEBUG: Updated self.board_cards from self.round.board to: {self.board_cards}",
                            **log_extra,
                        )
                    except Exception as e:
                        log_error(
                            f"Error reading or converting self.round.board: {e}",
                            **log_extra,
                        )
                        self.board_cards = []

                log_info(
                    f"New Street: {self.current_street}. Board string: '{self._get_current_board_str()}'",
                    **log_extra,
                )
                if action_type:
                    self._record_action(action_type=action_type)

            elif out_id == RoundPublicOutId.SMALLBLIND:
                amount = kwargs.get("amount", 0)
                if player_id and player:
                    self._record_action(
                        action_type=self.ACTION_POST_SB,
                        player_id=player_id,
                        action_amount=amount,
                    )
            elif out_id == RoundPublicOutId.BIGBLIND:
                amount = kwargs.get("amount", 0)
                if player_id and player:
                    self._record_action(
                        action_type=self.ACTION_POST_BB,
                        player_id=player_id,
                        action_amount=amount,
                    )

            # --- Player Actions (Pass Temperature) ---
            elif out_id == RoundPublicOutId.PLAYERCHECK:
                if player_id:
                    temp = self._last_action_temperature
                    self._record_action(
                        action_type=self.ACTION_CHECK,
                        player_id=player_id,
                        action_amount=0,
                        temperature=temp,
                    )
                    self._last_action_temperature = None  # Reset temp
            elif out_id == RoundPublicOutId.PLAYERCALL:
                call_amount = kwargs.get("paid_amount", 0)
                if player_id:
                    temp = self._last_action_temperature
                    self._record_action(
                        action_type=self.ACTION_CALL,
                        player_id=player_id,
                        action_amount=call_amount,
                        temperature=temp,
                    )
                    self._last_action_temperature = None  # Reset temp
            elif out_id == RoundPublicOutId.PLAYERFOLD:
                if player_id:
                    log_info(
                        f"DEBUG: Handling PLAYERFOLD for P{player_id}", **log_extra
                    )
                    temp = self._last_action_temperature
                    self._record_action(
                        action_type=self.ACTION_FOLD,
                        player_id=player_id,
                        action_amount=0,
                        temperature=temp,
                    )
                    self._last_action_temperature = None  # Reset temp

                    if (
                        self.round
                        and hasattr(self.round, "players")
                        and not self.winner_declared_this_round
                    ):
                        active_players = [
                            p
                            for p in self.round.players
                            if p and not getattr(p, "is_folded", True)
                        ]
                        log_info(
                            f"DEBUG: Active players after fold: {[p.id for p in active_players]}",
                            **log_extra,
                        )
                        if len(active_players) == 1:
                            winner = active_players[0]
                            winner_id = winner.id
                            total_pot = sum(getattr(self.round, "pot_size", [0]))
                            log_warning(
                                f"MANUAL WINNER DETECTION: Only P{winner_id} ({winner.name}) remains active after fold. Awarding pot of ${total_pot}.",
                                **log_extra,
                            )
                            self._record_action(
                                action_type=self.ACTION_WINS_POT,
                                player_id=winner_id,
                                amount_won=total_pot,
                                final_hand_rank="Premature Win",
                            )
                            self.winner_declared_this_round = True
                            self.round_over_flag = True
                            log_info(
                                "Setting round_over_flag = True due to manual winner detection.",
                                **log_extra,
                            )
            elif out_id == RoundPublicOutId.PLAYERRAISE:
                raise_by = kwargs.get("raised_by", 0)
                action_type = (
                    self.ACTION_BET if not self.street_has_bet else self.ACTION_RAISE
                )
                if player_id:
                    temp = self._last_action_temperature
                    self._record_action(
                        action_type=action_type,
                        player_id=player_id,
                        action_amount=raise_by,
                        temperature=temp,
                    )
                    self._last_action_temperature = None  # Reset temp
                    self.street_has_bet = True
            elif out_id == RoundPublicOutId.PLAYERWENTALLIN:
                pass

            # --- Action Required ---
            elif out_id == RoundPublicOutId.PLAYERACTIONREQUIRED:
                self._current_action_player_id = player_id
                self.to_call = kwargs.get("to_call", 0)
                self.can_check = self.to_call == 0
                self.can_raise = (
                    player and hasattr(player, "money") and player.money > self.to_call
                )
                self.can_fold = True
                self.min_raise = kwargs.get("min_raise", self.big_blind)
                self.pot_at_action_start = (
                    sum(getattr(self.round, "pot_size", [0])) if self.round else 0
                )
                log_info(
                    f"Action required for P{player_id}. To Call: {self.to_call}. Pot Before: {self.pot_at_action_start}. MinRaiseBy: {self.min_raise}",
                    **log_extra,
                )

            # --- Showdown / Winner Declaration ---
            elif out_id == RoundPublicOutId.PUBLICCARDSHOW:
                pass
            elif out_id == RoundPublicOutId.DECLAREPREMATUREWINNER:
                log_info(
                    f"DEBUG: Entered DECLAREPREMATUREWINNER handler. Kwargs: {kwargs}",
                    **log_extra,
                )
                if not self.winner_declared_this_round:
                    money_won = kwargs.get("money_won", 0)
                    if player_id:
                        self._record_action(
                            action_type=self.ACTION_WINS_POT,
                            player_id=player_id,
                            amount_won=money_won,
                        )
                        log_info(
                            f"Premature Winner (Event): P{player_id} wins ${money_won}",
                            **log_extra,
                        )
                        self.winner_declared_this_round = True
                else:
                    log_info(
                        "Skipping DECLAREPREMATUREWINNER event log as winner already declared manually.",
                        **log_extra,
                    )

            elif out_id == RoundPublicOutId.DECLAREFINISHEDWINNER:
                log_info(
                    f"DEBUG: Entered DECLAREFINISHEDWINNER handler. Kwargs: {kwargs}",
                    **log_extra,
                )
                if not self.winner_declared_this_round:
                    money_won = kwargs.get("money_won", 0)
                    hand_enum = kwargs.get("handname")
                    hand_name_str = format_hand_enum(hand_enum) if hand_enum else None
                    if player_id:
                        self._record_action(
                            action_type=self.ACTION_WINS_POT,
                            player_id=player_id,
                            amount_won=money_won,
                            final_hand_rank=hand_name_str,
                        )
                        log_info(
                            f"Finished Winner (Event): P{player_id} wins ${money_won} with {hand_name_str}",
                            **log_extra,
                        )
                        self.winner_declared_this_round = True
                else:
                    log_info(
                        "Skipping DECLAREFINISHEDWINNER event log as winner already declared manually.",
                        **log_extra,
                    )

            # --- Round End ---
            elif out_id == RoundPublicOutId.ROUNDFINISHED:
                log_info(
                    f"Round {self.round_num} Finished (ROUNDFINISHED Event).",
                    **log_extra,
                )
                self.round_over_flag = True
                if not self.winner_declared_this_round:
                    log_warning(
                        f"ROUNDFINISHED received but no winner was declared via event or manually for Hand ID {self.current_hand_id}.",
                        **log_extra,
                    )
                self._finalize_hand_data()

        # --- Handle Table Events ---
        elif isinstance(out_id, TablePublicOutId):
            if out_id == TablePublicOutId.PLAYERREMOVED:
                log_info(f"Player P{player_id} removed from table.", **log_extra)
            elif out_id == TablePublicOutId.ROUNDNOTINITIALIZED:
                log_error("Round not initialized event received.", **log_extra)

        # --- Handle ROUNDFINISHED if round object disappeared ---
        elif is_round_event and out_id == RoundPublicOutId.ROUNDFINISHED:
            log_info(
                f"Round {self.round_num} Finished (ROUNDFINISHED Event, Round object was None).",
                **log_extra,
            )
            if not self.round_over_flag:
                self.round_over_flag = True
            if not self.winner_declared_this_round:
                log_warning(
                    f"ROUNDFINISHED received (round was None) but no winner was declared for Hand ID {self.current_hand_id}.",
                    **log_extra,
                )
            self._finalize_hand_data()

    def privateOut(self, player_id, out_id, **kwargs):
        if isinstance(out_id, RoundPrivateOutId):
            if out_id == RoundPrivateOutId.DEALTCARDS:
                cards_raw = kwargs.get("cards")
                if (
                    cards_raw
                    and isinstance(cards_raw, (list, tuple))
                    and len(cards_raw) == 2
                ):
                    cards_tuples = tuple(
                        tuple(c)
                        for c in cards_raw
                        if isinstance(c, (list, tuple)) and len(c) == 2
                    )
                    if len(cards_tuples) == 2:
                        self._player_cards[player_id] = cards_tuples
                        self._record_action(
                            action_type=self.ACTION_DEALT_HOLE, player_id=player_id
                        )
                        card_str_log = format_cards_for_ai(cards_tuples)
                        log_info(
                            f"Dealt cards {card_str_log} to P{player_id}",
                            self.game_num,
                            self.round_num,
                        )
                    else:
                        log_error(
                            f"Card data conversion error for P{player_id}.",
                            self.game_num,
                            self.round_num,
                        )
                else:
                    log_error(
                        f"Invalid card data received for P{player_id}: {cards_raw}",
                        self.game_num,
                        self.round_num,
                    )

    def get_game_data(self):
        return self.game_data

    def write_game_data_to_csv(self, filepath):
        log_info(
            f"Writing game data for game {self.game_num} to {filepath}", self.game_num
        )
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=self.CSV_HEADERS, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(self.game_data)
            log_info(
                f"Successfully wrote {len(self.game_data)} rows to {filepath}",
                self.game_num,
            )
        except IOError as e:
            log_error(
                f"Error writing CSV file {filepath}: {e}", self.game_num, exc_info=True
            )
        except Exception as e:
            log_error(
                f"Unexpected error writing CSV {filepath}: {e}",
                self.game_num,
                exc_info=True,
            )


# ==============================================================================
# AI Action Logic (Unchanged)
# ==============================================================================
def get_ai_action(table_state: SimulationTable, player_id):
    player_name = table_state._get_player_name(player_id)
    model_name = table_state.ai_model_assignments.get(player_id)
    game_num = table_state.game_num
    round_num = table_state.round_num

    if not model_name:
        log_error(
            f"No AI model assigned to P{player_id} ({player_name}). Defaulting to FOLD.",
            game_num,
            round_num,
        )
        return RoundPublicInId.FOLD, {}, AI_TEMPERATURE

    if VERBOSE:
        print(f"--- AI Thinking ({player_name} using {model_name}) ---")

    prompt = format_state_for_ai(table_state, player_id, game_num, round_num)
    if "Error:" in prompt:
        log_error(
            f"Error formatting state for AI P{player_id}: {prompt}. Defaulting to FOLD.",
            game_num,
            round_num,
        )
        return RoundPublicInId.FOLD, {}, AI_TEMPERATURE

    history = table_state.ai_message_history[round_num][player_id]
    system_prompt = {
        "role": "system",
        "content": "You are a Texas Hold'em poker AI. Analyze the state and decide your action. Respond ONLY with the action name (FOLD, CHECK, CALL, RAISE). If RAISING, add ' AMOUNT: X' on the SAME line, where X is the integer amount to raise BY (additional chips on top of the call amount).",
    }
    messages = [system_prompt] + history + [{"role": "user", "content": prompt}]

    log_info(
        f"--- AI Turn: P{player_id} ({player_name}, Model: {model_name}) ---",
        game_num,
        round_num,
    )
    if VERBOSE:
        log_info(f"Prompt Sent:\n{prompt}", game_num, round_num)

    ai_response_text = query_together_ai(
        model_name, messages, AI_TEMPERATURE, game_num, round_num
    )
    log_info(
        f"Raw Response: {ai_response_text or '<< No Response >>'}", game_num, round_num
    )

    player = table_state.seats.getPlayerById(player_id)
    if not player or not hasattr(player, "money"):
        log_error(
            f"AI Action: Could not find player object or money for ID {player_id}. Defaulting to FOLD.",
            game_num,
            round_num,
        )
        return RoundPublicInId.FOLD, {}, AI_TEMPERATURE

    action_enum, action_kwargs = parse_ai_action(ai_response_text, game_num, round_num)

    validated_action_enum = action_enum
    validated_action_kwargs = action_kwargs
    needs_fallback = False
    warn_msg = ""

    if action_enum is None:
        needs_fallback = True
        warn_msg = f"AI action parsing failed or invalid raise amount from '{ai_response_text}'."
    else:
        if action_enum == RoundPublicInId.FOLD:
            if not table_state.can_fold:
                needs_fallback = True
                warn_msg = "AI chose FOLD when not possible (rare)."
        elif action_enum == RoundPublicInId.CHECK:
            if not table_state.can_check:
                needs_fallback = True
                warn_msg = f"AI chose CHECK when not possible (To Call: ${table_state.to_call})."
        elif action_enum == RoundPublicInId.CALL:
            if table_state.can_check:
                needs_fallback = True
                warn_msg = f"AI chose CALL when CHECK was possible."
            elif table_state.to_call <= 0:
                needs_fallback = True
                warn_msg = f"AI chose CALL when to_call was ${table_state.to_call}."
            elif player.money <= 0:
                needs_fallback = True
                warn_msg = f"AI chose CALL with zero stack."
        elif action_enum == RoundPublicInId.RAISE:
            if not table_state.can_raise:
                needs_fallback = True
                warn_msg = f"AI chose RAISE when not allowed."
            else:
                raise_by = validated_action_kwargs.get("raise_by", 0)
                max_raise_by = player.money - table_state.to_call
                min_raise_by_required = table_state.min_raise
                is_all_in_raise = (table_state.to_call + raise_by) >= player.money

                if raise_by <= 0:
                    needs_fallback = True
                    warn_msg = f"AI chose RAISE with invalid amount {raise_by} (<=0)."
                elif raise_by > max_raise_by:
                    log_warning(
                        f"AI chose RAISE {raise_by}, exceeding max possible ({max_raise_by}). Correcting to all-in raise.",
                        game_num,
                        round_num,
                    )
                    validated_action_kwargs["raise_by"] = max_raise_by
                elif raise_by < min_raise_by_required and not is_all_in_raise:
                    needs_fallback = True
                    warn_msg = f"AI chose RAISE {raise_by}, below minimum ({min_raise_by_required}) and not all-in."

    if needs_fallback:
        log_warning(
            f"AI Action Invalid/Failed: {warn_msg} Player: {player_name}. Applying fallback.",
            game_num,
            round_num,
        )
        if table_state.can_check:
            fallback_msg = "Fallback Action: CHECK"
            validated_action_enum = RoundPublicInId.CHECK
            validated_action_kwargs = {}
        elif table_state.to_call > 0 and player.money > 0:
            fallback_msg = "Fallback Action: CALL"
            validated_action_enum = RoundPublicInId.CALL
            validated_action_kwargs = {}
        else:
            fallback_msg = "Fallback Action: FOLD"
            validated_action_enum = RoundPublicInId.FOLD
            validated_action_kwargs = {}
        log_info(fallback_msg, game_num, round_num)

    if len(history) > 10:
        history = history[-10:]
    history.append({"role": "user", "content": prompt})
    assistant_response_content = f"{validated_action_enum.name}"
    if validated_action_enum == RoundPublicInId.RAISE:
        assistant_response_content += (
            f" AMOUNT: {validated_action_kwargs.get('raise_by', 0)}"
        )
    history.append({"role": "assistant", "content": assistant_response_content})

    log_info(
        f"Validated Action: {validated_action_enum.name} {validated_action_kwargs}",
        game_num,
        round_num,
    )
    # Return temp used for decision, even if action was fallback
    return validated_action_enum, validated_action_kwargs, AI_TEMPERATURE


# ==============================================================================
# Main Simulation Execution Block (Corrected Temp Handling)
# ==============================================================================
if __name__ == "__main__":
    temps = [1.0]
    for temp in temps:
        AI_TEMPERATURE = temp
        OUTPUT_DIR = f"simulation_{temp}"
        start_time_total = time.time()
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Starting simulation: {GAMES} games, {ROUNDS} rounds each.")
        print(f"Output directory: {output_path.resolve()}")
        print(f"Logging to: {LOG_FILE_NAME}")
        log_info(
            f"Starting simulation: {GAMES} games, {ROUNDS} rounds each. Output Dir: {output_path.resolve()}"
        )

        for game_num in range(1, GAMES + 1):
            start_time_game = time.time()
            print(f"\n--- Starting Game {game_num}/{GAMES} ---")
            log_info(f"--- Starting Game {game_num}/{GAMES} ---", game_num=game_num)

            table = SimulationTable(
                game_num=game_num,
                _id=game_num,
                seats=PlayerSeats([None] * NUM_PLAYERS),
                buyin=BUYIN,
                small_blind=SMALL_BLIND,
                big_blind=BIG_BLIND,
            )
            log_info(
                f"Table initialized. Buy-in: {BUYIN}, Blinds: {SMALL_BLIND}/{BIG_BLIND}.",
                game_num,
            )

            players = []
            player_id_counter = 1
            temp_ai_assignments = {}
            for i in range(NUM_PLAYERS):
                ai_player_id = player_id_counter
                player_id_counter += 1
                model_index = i % len(AI_MODEL_LIST)
                model_full_name = AI_MODEL_LIST[model_index]
                ai_short_name = AI_MODEL_SHORT_NAMES.get(
                    model_full_name, f"AI_{ai_player_id}"
                )
                ai_player_name = f"{ai_short_name}"
                ai_player = Player(
                    table_id=table.id, _id=ai_player_id, name=ai_player_name, money=BUYIN
                )
                players.append(ai_player)
                temp_ai_assignments[ai_player_id] = model_full_name
                log_info(
                    f"Created AI Player: {ai_player_name} (ID: {ai_player_id}) with ${BUYIN}. Model: {model_full_name}",
                    game_num,
                )

            random.shuffle(players)
            log_info(f"Player seating order: {[p.name for p in players]}", game_num)
            for p in players:
                try:
                    table.publicIn(p.id, TablePublicInId.BUYIN, player=p)
                    log_info(f"Seated player {p.name} (ID: {p.id})", game_num)
                except Exception as e:
                    log_error(
                        f"Error seating player {p.name} (ID: {p.id}): {e}",
                        game_num,
                        exc_info=True,
                    )
            time.sleep(0.1)

            joined_players = table.seats.getPlayerGroup()
            if len(joined_players) < 2:
                log_error(
                    f"Game {game_num} aborted: Only {len(joined_players)} players joined.",
                    game_num,
                )
                print(
                    f"Error: Game {game_num} aborted due to insufficient players joining."
                )
                continue

            table.ai_model_assignments = temp_ai_assignments
            table.assign_ai_models()

            for round_num_in_game in range(1, ROUNDS + 1):
                log_extra = {"game_num": game_num, "round_num": round_num_in_game}
                log_info(
                    f"--- Starting Round {round_num_in_game}/{ROUNDS} ---", **log_extra
                )

                active_players_obj = table.seats.getPlayerGroup()
                active_players_with_money = [
                    p
                    for p in active_players_obj
                    if p and hasattr(p, "money") and p.money > 0
                ]

                if len(active_players_with_money) < 2:
                    log_warning(
                        f"Not enough players with money ({len(active_players_with_money)}) to start round {round_num_in_game}. Ending game early.",
                        **log_extra,
                    )
                    break

                initiator_id = active_players_with_money[0].id
                log_info(f"Attempting STARTROUND initiated by P{initiator_id}", **log_extra)

                try:
                    table.publicIn(
                        initiator_id, TablePublicInId.STARTROUND, round_id=round_num_in_game
                    )
                except Exception as e:
                    log_error(
                        f"Error sending STARTROUND for round {round_num_in_game}: {e}",
                        exc_info=True,
                        **log_extra,
                    )
                    break

                if not table.round:
                    log_error(
                        f"Round {round_num_in_game} failed to initialize after STARTROUND signal. Ending game.",
                        **log_extra,
                    )
                    break

                round_action_start_time = time.time()
                action_timeout_seconds = AI_REQUEST_TIMEOUT * NUM_PLAYERS * 6

                while table.round and not table.round_over_flag:
                    if time.time() - round_action_start_time > action_timeout_seconds:
                        log_error(
                            f"Round {round_num_in_game} timed out after {action_timeout_seconds}s. Forcing end.",
                            **log_extra,
                        )
                        table.round_over_flag = True
                        if table.current_hand_actions:
                            log_warning(
                                "Attempting to finalize hand data after timeout.",
                                **log_extra,
                            )
                            table._finalize_hand_data()
                        break

                    action_player_id_to_process = table._current_action_player_id
                    if action_player_id_to_process:
                        table._current_action_player_id = None

                        player = table.seats.getPlayerById(action_player_id_to_process)
                        # if not player:
                        #     log_warning(
                        #         f"Action requested for invalid Player ID {action_player_id_to_process}. Skipping.",
                        #         **log_extra,
                        #     )
                        #     continue
                        # if player.is_folded or player.is_all_in:
                        #     log_info(
                        #         f"Action requested for player P{player.id} who is folded or all-in. Skipping turn.",
                        #         **log_extra,
                        #     )
                        #     time.sleep(0.05)
                        #     continue

                        current_player_obj_from_lib = getattr(
                            table.round, "current_player", None
                        )
                        if (
                            not current_player_obj_from_lib
                            or player.id != current_player_obj_from_lib.id
                        ):
                            lib_player_id = getattr(
                                current_player_obj_from_lib, "id", "None"
                            )
                            log_warning(
                                f"Action sync issue R{round_num_in_game}. Flag={action_player_id_to_process}, Lib={lib_player_id}. Trusting flag.",
                                **log_extra,
                            )

                        try:
                            action_enum, action_kwargs, temp_used = get_ai_action(
                                table, player.id
                            )

                            # --- Store temperature before calling publicIn ---
                            table._last_action_temperature = temp_used

                            # --- Robust publicIn Call (without temperature in kwargs) ---
                            try:
                                log_info(
                                    f"DEBUG: Sending action {action_enum.name} {action_kwargs} for P{player.id} to table.publicIn",
                                    **log_extra,
                                )
                                # DO NOT pass temp_used here: table.publicIn(player.id, action_enum, **action_kwargs, temperature=temp_used) # WRONG
                                table.publicIn(
                                    player.id, action_enum, **action_kwargs
                                )  # CORRECTED
                                log_info(
                                    f"DEBUG: table.publicIn for P{player.id} action completed without error.",
                                    **log_extra,
                                )
                            except Exception as publicIn_e:
                                log_error(
                                    f"CRITICAL: Exception during table.publicIn for P{player.id} action {action_enum.name}: {publicIn_e}",
                                    exc_info=True,
                                    **log_extra,
                                )
                                log_warning(
                                    f"Forcing FOLD for player P{player.id} due to publicIn error.",
                                    **log_extra,
                                )
                                try:
                                    table.publicIn(player.id, RoundPublicInId.FOLD)
                                except Exception as fold_e:
                                    log_error(
                                        f"Error forcing fold for P{player.id} after publicIn error: {fold_e}",
                                        **log_extra,
                                    )
                                table.round_over_flag = True
                                break  # Exit action loop
                            # --- End robust publicIn Call ---

                        except Exception as e:
                            log_error(
                                f"Error during AI action cycle (get_ai_action?) for P{player.id}: {e}",
                                exc_info=True,
                                **log_extra,
                            )
                            log_warning(
                                f"Forcing FOLD for player P{player.id} due to action cycle error.",
                                **log_extra,
                            )
                            try:
                                table.publicIn(player.id, RoundPublicInId.FOLD)
                            except Exception as fold_e:
                                log_error(
                                    f"Error forcing fold for P{player.id} after AI cycle error: {fold_e}",
                                    **log_extra,
                                )

                    time.sleep(0.05)
                # --- End of Round Action Loop ---

                if table.round_over_flag:
                    log_info(f"--- ROUND {round_num_in_game} END ---", **log_extra)
                    final_players = table.seats.getPlayerGroup()
                    log_stacks = [f"Stacks after Round {round_num_in_game}:"]
                    if final_players:
                        final_players.sort(key=lambda p: p.id if p else 0)
                        for p in final_players:
                            if p:
                                log_stacks.append(
                                    f"  - {p.name} (ID:{p.id}): ${getattr(p, 'money', 'N/A')}"
                                )
                    else:
                        log_stacks.append("  No players remaining.")
                    log_info("\n".join(log_stacks), **log_extra)
                    if table.round:
                        table.round = None

                else:  # Round loop ended unexpectedly
                    log_warning(
                        f"Round {round_num_in_game} loop ended unexpectedly (round_over_flag is False).",
                        **log_extra,
                    )
                    if (
                        not table.winner_declared_this_round
                        and table.round
                        and hasattr(table.round, "players")
                    ):
                        active_players = [
                            p
                            for p in table.round.players
                            if p and not getattr(p, "is_folded", True)
                        ]
                        if len(active_players) == 1:
                            winner = active_players[0]
                            winner_id = winner.id
                            total_pot = sum(getattr(table.round, "pot_size", [0]))
                            log_warning(
                                f"MANUAL WINNER DETECTION (End of Loop): Only P{winner_id} remains. Awarding pot ${total_pot}.",
                                **log_extra,
                            )
                            table._record_action(
                                action_type=table.ACTION_WINS_POT,
                                player_id=winner_id,
                                amount_won=total_pot,
                                final_hand_rank="Premature Win",
                            )
                            table.winner_declared_this_round = True
                        elif len(active_players) > 1:
                            log_warning(
                                f"Round loop ended unexpectedly with multiple ({len(active_players)}) players still active. Hand result unknown.",
                                **log_extra,
                            )
                        else:
                            log_warning(
                                f"Round loop ended unexpectedly with no active players? Hand result unknown.",
                                **log_extra,
                            )

                    if table.current_hand_actions:
                        log_warning(
                            "Attempting to finalize partial hand data after unexpected loop end.",
                            **log_extra,
                        )
                        table._finalize_hand_data()
                    if table.round:
                        table.round = None
                    break  # Exit game loop

            # --- End of Round Loop ---

            game_csv_path = output_path / f"game_{game_num}.csv"
            table.write_game_data_to_csv(game_csv_path)

            end_time_game = time.time()
            duration_game = end_time_game - start_time_game
            print(f"--- Game {game_num} Finished ({duration_game:.2f}s) ---")
            log_info(f"--- Game {game_num} Finished ({duration_game:.2f}s) ---", game_num)

        # --- End of Game Loop ---

        end_time_total = time.time()
        duration_total = end_time_total - start_time_total
        print(f"\n=== Simulation Finished ===")
        print(f"Total Games: {GAMES}")
        print(f"Rounds per Game: {ROUNDS}")
        print(f"Total Time: {duration_total:.2f} seconds")
        print(f"Detailed CSV logs saved in: {output_path.resolve()}")
        print(f"Detailed runtime logs saved in: {LOG_FILE_NAME}")
        log_info(f"=== Simulation Finished ===")
        log_info(
            f"Total Games: {GAMES}, Rounds per Game: {ROUNDS}, Total Time: {duration_total:.2f}s"
        )
