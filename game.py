# -*- coding: utf-8 -*-
"""
Plays a game of Texas Hold'em poker, allowing for Human vs AI or AI vs AI modes.
Uses the 'pokerlib' library for game logic and the Together AI API for AI opponents.
Integrates 'deuces' library for hand evaluation and win probability estimation.
"""

import logging
import os
import random
import sys
import time
import traceback
from collections import defaultdict, deque

import requests
from dotenv import load_dotenv
# Necessary imports from pokerlib
from pokerlib import Player, PlayerSeats, Table
from pokerlib.enums import (Hand, Rank, RoundPrivateOutId, RoundPublicInId,
                            RoundPublicOutId, Suit, TablePrivateOutId,
                            TablePublicInId, TablePublicOutId, Turn)

# --- Deuces Integration ---
try:
    from deuces import Card, Deck, Evaluator
except ImportError:
    print("Error: 'deuces' library not found. Please install it: pip install deuces")
    sys.exit(1)
# --------------------------

# ==============================================================================
# Constants and Configuration
# ==============================================================================

# --- General Settings ---
VERBOSE = False             # If True, print extra AI debugging information.
ALWAYS_CONTINUE = False     # If True, automatically start the next round without asking.
AI_ONLY_MODE = True        # <<< SET TO True FOR ALL AI, False FOR HUMAN vs AI >>>
CLEAR_SCREEN = True         # <<< CHANGE to False to prevent clearing terminal at round end >>>
HUMAN_PLAYER_ID = 1         # ID of the human player if AI_ONLY_MODE is False. Must be > 0.
HUMAN_PLAYER_NAME = "Aum"   # <<< Name for the human player >>>
LOG_FILE_NAME = 'ai_poker_log.txt' # File to log AI interactions and game events.

# --- Probability Display Settings ---
SHOW_PROBABILITIES = True           # <<< SET TO True to display win probabilities, False to hide >>>
PROBABILITY_SIMULATIONS = 5000     # <<< Number of Monte Carlo simulations for probability (higher = slower but more accurate) >>>

# --- Game Parameters ---
BUYIN = 1000                # Default starting money for each player.
SMALL_BLIND = 5             # Small blind amount.
BIG_BLIND = 10              # Big blind amount.

# --- AI Configuration ---
AI_MODEL_LIST = [           # Models for AI opponents (requires Together AI API Key).
    # 'mistralai/Mistral-7B-Instruct-v0.3',
    # 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    # 'upstage/SOLAR-10.7B-Instruct-v1.0',
    # 'Gryphe/MythoMax-L2-13b',
    # 'mistralai/Mistral-Small-24B-Instruct-2501',
    # 'google/gemma-2-27b-it',
    # 'Qwen/QwQ-32B-Preview',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'Qwen/Qwen2.5-72B-Instruct-Turbo',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    # 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    # 'microsoft/WizardLM-2-8x22B',
    # 'databricks/dbrx-instruct',
    # 'deepseek-ai/DeepSeek-V3',
    # 'deepseek-ai/DeepSeek-R1',
]
# Dictionary mapping full model names to desired short names for display.
AI_MODEL_SHORT_NAMES = {
    'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral v0.3 7B',
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': 'Llama 3.1 8B',
    'upstage/SOLAR-10.7B-Instruct-v1.0': 'Upstage Solar 11B',
    'Gryphe/MythoMax-L2-13b': 'Gryphe MythoMax-L2 13B',
    'mistralai/Mistral-Small-24B-Instruct-2501': 'Mistral Small 3 24B',
    'google/gemma-2-27b-it': 'Gemma 2 27B',
    'Qwen/QwQ-32B-Preview': 'Qwen QwQ 32B',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo': 'Llama 3.3 70B',
    'Qwen/Qwen2.5-72B-Instruct-Turbo': 'Qwen 2.5 72B',
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 'Mixtral 8x7B',
    'mistralai/Mixtral-8x22B-Instruct-v0.1': 'Mixtral 8x22B',
    'microsoft/WizardLM-2-8x22B': 'WizardLM-2 8x22B',
    'databricks/dbrx-instruct': 'DBRX Instruct',
    'deepseek-ai/DeepSeek-V3': 'DeepSeek V3',
    'deepseek-ai/DeepSeek-R1': 'DeepSeek R1',
    # Add mappings for any other models used
}

AI_TEMPERATURE = 1.0       # Sampling temperature for AI model responses.
AI_REQUEST_TIMEOUT = 60     # Timeout in seconds for API requests to Together AI.
AI_RETRY_DELAY = 7          # Delay in seconds before retrying a failed API request.

# --- Derived Constants & Initial Checks ---
NUM_AI_MODELS = len(AI_MODEL_LIST)
NUM_PLAYERS = NUM_AI_MODELS if AI_ONLY_MODE else NUM_AI_MODELS + 1

if NUM_PLAYERS < 2:
    print(f"Error: Need at least 2 total players (configured {NUM_PLAYERS}).")
    sys.exit(1)
if NUM_AI_MODELS == 0 and (AI_ONLY_MODE or not AI_ONLY_MODE and NUM_PLAYERS > 1):
    print("Error: AI players required but AI_MODEL_LIST is empty.")
    sys.exit(1)
if not AI_ONLY_MODE and HUMAN_PLAYER_ID <= 0:
    print(f"Error: HUMAN_PLAYER_ID must be positive. Found: {HUMAN_PLAYER_ID}")
    sys.exit(1)

# --- ANSI Color Codes ---
class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = "\033[0m"; BOLD = "\033[1m"; UNDERLINE = "\033[4m"
    BLACK = "\033[30m"; RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"; BRIGHT_RED = "\033[91m"; BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"; BRIGHT_BLUE = "\033[94m"; BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"; BRIGHT_WHITE = "\033[97m"
    BG_BLACK = "\033[40m"; BG_RED = "\033[41m"; BG_GREEN = "\033[42m"; BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"; BG_MAGENTA = "\033[45m"; BG_CYAN = "\033[46m"; BG_WHITE = "\033[47m"

# --- Card Formatting Maps ---
RANK_MAP_POKERLIB = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5", Rank.SIX: "6",
    Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9", Rank.TEN: "T", Rank.JACK: "J",
    Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A"
}
SUIT_MAP_POKERLIB = { Suit.SPADE: "♠", Suit.CLUB: "♣", Suit.DIAMOND: "♦", Suit.HEART: "♥" }
SUIT_COLOR_MAP = {
    Suit.SPADE: Colors.WHITE,
    Suit.CLUB: Colors.WHITE,
    Suit.DIAMOND: Colors.BRIGHT_RED,
    Suit.HEART: Colors.BRIGHT_RED
}

# --- Mappings for Deuces Card Conversion ---
RANK_MAP_TO_DEUCES = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5", Rank.SIX: "6",
    Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9", Rank.TEN: "T", Rank.JACK: "J",
    Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A"
}
SUIT_MAP_TO_DEUCES = {
    Suit.SPADE: "s", Suit.CLUB: "c", Suit.DIAMOND: "d", Suit.HEART: "h"
}
# --------------------------------------------


# ==============================================================================
# Logging Setup
# ==============================================================================

# Clear log file if it exists
if os.path.exists(LOG_FILE_NAME):
    os.remove(LOG_FILE_NAME)

# Configure logger for AI interactions and game events
ai_logger = logging.getLogger('AIPokerLog')
ai_logger.setLevel(logging.INFO)
ai_logger.propagate = False # Prevent passing logs to root logger

# Create file handler
file_handler = logging.FileHandler(LOG_FILE_NAME)
file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
ai_logger.addHandler(file_handler)

ai_logger.info("--- AI Poker Log Initialized ---")

# ==============================================================================
# API Key Loading
# ==============================================================================

load_dotenv()  # Load environment variables from .env file
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")

def colorize(text, color):
    """Applies ANSI color codes to text."""
    return f"{color}{text}{Colors.RESET}"

# Check for API key only if AI players are actually needed
if (AI_ONLY_MODE or NUM_AI_MODELS > 0) and not TOGETHER_AI_API_KEY:
    error_msg = "ERROR: AI players enabled, but TOGETHER_AI_API_KEY environment variable is missing or empty."
    print(colorize(error_msg, Colors.RED))
    ai_logger.error(error_msg)
    sys.exit(1)
elif (AI_ONLY_MODE or NUM_AI_MODELS > 0):
    ai_logger.info("TOGETHER_AI_API_KEY loaded successfully.")

# ==============================================================================
# Helper Functions
# ==============================================================================

def format_card_terminal(card):
    """Formats a card tuple (Rank, Suit) into a short, colored string for terminal display.

    Args:
        card: A tuple representing a card, e.g., (Rank.ACE, Suit.SPADE).

    Returns:
        A colored string representation of the card (e.g., "A♠") or "??" if invalid.
    """
    fallback_card = colorize("??", Colors.BRIGHT_BLACK)
    if not card or not hasattr(card, '__len__') or len(card) != 2:
        return fallback_card
    try:
        rank, suit = card
        rank_str = RANK_MAP_POKERLIB.get(rank)
        suit_str = SUIT_MAP_POKERLIB.get(suit)
        if rank_str is None or suit_str is None:
            return fallback_card

        color = SUIT_COLOR_MAP.get(suit, Colors.WHITE)
        card_text = f"{rank_str}{suit_str}"
        # Bold face cards
        if rank >= Rank.JACK:
            card_text = Colors.BOLD + card_text
        return colorize(card_text, color)
    except (TypeError, ValueError, IndexError):
        # Catch potential issues if card structure is unexpected
        return fallback_card

def format_cards_terminal(cards):
    """Formats an iterable of cards into a space-separated colored string for the terminal.

    Args:
        cards: An iterable (list, tuple) of card tuples.

    Returns:
        A space-separated string of formatted cards, or an empty string if input is invalid.
    """
    if hasattr(cards, '__iter__'):
        return " ".join(format_card_terminal(c) for c in cards)
    return ""

def format_card_for_ai(card):
    """Formats a card tuple into a simple string for AI prompts (e.g., "AS", "KH").

    Args:
        card: A tuple representing a card, e.g., (Rank.ACE, Suit.SPADE).

    Returns:
        A string representation (e.g., "A♠") or "??" if invalid. (Kept original suit symbol)
    """
    if not card:
        return "??"
    try:
        if hasattr(card, '__len__') and len(card) == 2:
            rank, suit = card
            rank_str = RANK_MAP_POKERLIB.get(rank)
            suit_str = SUIT_MAP_POKERLIB.get(suit)
            if rank_str is None or suit_str is None:
                 return "??"
            return f"{rank_str}{suit_str}"
        else:
            return "??"
    except (TypeError, KeyError, ValueError, IndexError):
        # Handles cases where card is not a valid tuple or rank/suit are invalid
        return "??"

def format_cards_for_ai(cards):
    """Formats an iterable of cards into a space-separated string for AI prompts.

    Args:
        cards: An iterable (list, tuple) of card tuples.

    Returns:
        A space-separated string of AI-formatted cards, or an empty string if input is invalid.
    """
    if hasattr(cards, '__iter__'):
        return " ".join(format_card_for_ai(c) for c in cards)
    return ""

def format_hand_enum(hand_enum):
    """Converts a pokerlib Hand enum member into a readable title-case string.

    Args:
        hand_enum: A member of the pokerlib.enums.Hand enum.

    Returns:
        A formatted string (e.g., "Straight Flush") or "Unknown Hand".
    """
    return hand_enum.name.replace("_", " ").title() if hand_enum else "Unknown Hand"

def clear_terminal():
    """Clears the terminal screen if CLEAR_SCREEN is enabled."""
    if CLEAR_SCREEN:
        os.system('cls' if os.name == 'nt' else 'clear')

# ==============================================================================
# AI Interaction Functions
# ==============================================================================

def query_together_ai(model_name, messages, temperature):
    """Sends a prompt to the Together AI API and returns the model's response.

    Handles API errors, timeouts, and retries.

    Args:
        model_name: The name of the model to query (e.g., 'mistralai/Mistral-7B-Instruct-v0.3').
        messages: A list of message dictionaries for the chat completion API
                  (following OpenAI's format: [{'role': 'user', 'content': '...'}, ...]).
        temperature: The sampling temperature for the generation.

    Returns:
        The content string from the AI's response, or None if an error occurred.
    """
    api_endpoint = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "PokerCLI-AI-Test"  # Optional: Identify your application
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 150  # Limit response length
    }
    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=AI_REQUEST_TIMEOUT)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            # Check if the response structure is as expected
            if (data and 'choices' in data and len(data['choices']) > 0 and
                    'message' in data['choices'][0] and 'content' in data['choices'][0]['message']):
                return data['choices'][0]['message']['content'].strip()
            else:
                warn_msg = f"W: Unexpected API response structure from {model_name}: {data}"
                print(colorize(warn_msg, Colors.YELLOW))
                ai_logger.warning(warn_msg)
                return None # Or potentially retry depending on the error

        except requests.exceptions.Timeout:
            warn_msg = f"W: API request to {model_name} timed out (attempt {attempt + 1}/{max_retries})."
            print(colorize(warn_msg, Colors.YELLOW))
            ai_logger.warning(warn_msg)
            attempt += 1
            if attempt >= max_retries:
                return None
            print(colorize(f"Retrying in {AI_RETRY_DELAY}s...", Colors.YELLOW))
            time.sleep(AI_RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            err_msg = f"Error querying {model_name}: {e}"
            # Include response text if available for more context
            if hasattr(e, 'response') and e.response is not None:
                err_msg += f" | Response Status: {e.response.status_code} | Response Body: {e.response.text}"
            print(colorize(err_msg, Colors.RED))
            ai_logger.error(err_msg)

            # Specific handling for rate limiting (429)
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                attempt += 1
                retry_delay = AI_RETRY_DELAY * (attempt + 1) # Exponential backoff might be better
                print(colorize(f"Rate limit hit. Retrying in {retry_delay}s (attempt {attempt}/{max_retries})...", Colors.YELLOW))
                time.sleep(retry_delay)
            else:
                return None # Don't retry for other request exceptions

        except Exception as e:
            # Catch any other unexpected errors during the API call
            err_msg = f"Unexpected error during API call to {model_name}: {e}"
            print(colorize(err_msg, Colors.RED))
            ai_logger.error(err_msg, exc_info=True) # Log traceback for unexpected errors
            return None

    # If loop completes without returning, all retries failed
    fail_msg = f"API call to {model_name} failed after {max_retries} retries."
    print(colorize(fail_msg, Colors.RED))
    ai_logger.error(fail_msg)
    return None

def parse_ai_action(response_text):
    """Parses the AI's text response to determine the poker action and raise amount.

    Looks for keywords like "FOLD", "CHECK", "CALL", "RAISE".
    If "RAISE" is found, attempts to extract the integer amount following it.

    Args:
        response_text: The raw string response from the AI model.

    Returns:
        A tuple containing:
        - action_enum: The corresponding RoundPublicInId enum member (e.g., RoundPublicInId.RAISE).
        - action_kwargs: A dictionary containing necessary arguments for the action
                         (e.g., {'raise_by': 20} for a raise, empty {} otherwise).
        Defaults to CHECK if parsing fails.
    """
    if not response_text:
        return RoundPublicInId.CHECK, {} # Default action if no response

    response_lines = response_text.lower().split('\n')
    action = None
    raise_by = 0
    action_line_found = False

    # Primary parsing: Check for action keywords at the start of any line
    for line in response_lines:
        line = line.strip()
        if line.startswith("fold"):
            action = RoundPublicInId.FOLD
            action_line_found = True
            break
        if line.startswith("check"):
            action = RoundPublicInId.CHECK
            action_line_found = True
            break
        if line.startswith("call"):
            action = RoundPublicInId.CALL
            action_line_found = True
            break
        if line.startswith("raise"):
            action = RoundPublicInId.RAISE
            action_line_found = True
            parts = line.split()
            # Try to find the raise amount after "raise", "amount:", or "by:"
            for i, part in enumerate(parts):
                 if part == "amount" and i + 1 < len(parts) and parts[i+1].strip(':').isdigit():
                     raise_by = int(parts[i+1].strip(':'))
                     break
                 elif part == "by" and i + 1 < len(parts) and parts[i+1].isdigit():
                     raise_by = int(parts[i+1])
                     break
                 # Check if a part immediately following "raise" is a digit
                 elif i > 0 and parts[i-1] == "raise" and part.strip(':').isdigit():
                     raise_by = int(part.strip(':'))
                     break
                 # Fallback: check if any part *is* a digit (less reliable)
                 elif part.strip(':').isdigit():
                     raise_by = int(part.strip(':'))
                     # Don't break immediately, prefer explicit "amount:" or "by:"
            break # Action found, stop searching lines

    # Secondary parsing: If no action keyword was found at the start, check for "action:" prefix
    if not action_line_found:
         for line in response_lines:
            line = line.strip()
            if line.startswith("action:"):
                line_content = line[len("action:"):].strip()
                if "raise" in line_content:
                    action = RoundPublicInId.RAISE
                    parts = line_content.split()
                    # Find raise amount within the rest of the line
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            raise_by = int(part)
                            break
                        elif part in ["by", "amount"] and i + 1 < len(parts) and parts[i+1].strip(':').isdigit():
                             raise_by = int(parts[i+1].strip(':'))
                             break
                    break # Action found
                elif "call" in line_content:
                    action = RoundPublicInId.CALL
                    break
                elif "check" in line_content:
                    action = RoundPublicInId.CHECK
                    break
                elif "fold" in line_content:
                    action = RoundPublicInId.FOLD
                    break

    # Validation and Fallback
    if action == RoundPublicInId.RAISE and raise_by <= 0:
        # If AI said RAISE but didn't provide a valid amount, default to CALL
        print(colorize(f"[AI WARN] AI responded RAISE but failed to specify a valid amount > 0. Defaulting to CALL.", Colors.YELLOW))
        ai_logger.warning(f"AI response parsed as RAISE but raise_by was {raise_by}. Raw: '{response_text}'. Defaulting to CALL.")
        action = RoundPublicInId.CALL
        raise_by = 0
    elif action is None:
        # If no action could be parsed at all, default to CHECK
        print(colorize(f"[AI WARN] Could not parse action from AI response: '{response_text}'. Defaulting to CHECK.", Colors.YELLOW))
        ai_logger.warning(f"Could not parse action from AI response: '{response_text}'. Defaulting to CHECK.")
        action = RoundPublicInId.CHECK
        raise_by = 0

    # Prepare return values
    kwargs = {'raise_by': raise_by} if action == RoundPublicInId.RAISE else {}
    return action, kwargs

def format_state_for_ai(table_state, player_id_acting):
    """Formats the current game state into a markdown string suitable for the AI prompt.

    Args:
        table_state: The CommandLineTable instance containing the game state.
        player_id_acting: The ID of the player whose turn it is.

    Returns:
        A formatted string describing the game state, or an error message string.
    """
    lines = []
    # --- Basic State Checks ---
    if not table_state.round or not hasattr(table_state.round, 'turn'):
        return "Error: Game state is missing round or turn information."
    if not hasattr(table_state, '_player_cards'):
         return "Error: Table state missing _player_cards information."

    try:
        # --- Round Information ---
        lines.append(f"## Poker Hand State - Round {table_state.round.id}")
        lines.append(f"**Current Stage:** {table_state.round.turn.name}")

        # --- Board Cards ---
        board_cards = []
        if hasattr(table_state.round, 'board'):
             board_cards = [tuple(c) for c in table_state.round.board if isinstance(c, (list,tuple)) and len(c)==2]
        lines.append(f"**Board:** [ {format_cards_for_ai(board_cards)} ] ({len(board_cards)} cards)")

        # --- Pot Size ---
        pot_total = 0
        if hasattr(table_state.round, 'pot_size') and isinstance(table_state.round.pot_size, list):
            pot_total = sum(table_state.round.pot_size)
        lines.append(f"**Total Pot:** ${pot_total}")

        # --- Player Information ---
        lines.append("\n**Players:** (Order is position relative to dealer)")
        acting_player_obj = None
        max_name_len = 0
        players_in_round = []
        if hasattr(table_state.round, 'players') and table_state.round.players:
            players_in_round = table_state.round.players
            try:
                # Calculate max name length for alignment, handle potential missing names
                max_name_len = max(len(p.name if hasattr(p, 'name') else f'P{p.id}') for p in players_in_round if p and hasattr(p, 'id'))
            except ValueError: # Handle case where players_in_round might be empty or contain invalid players
                max_name_len = 10 # Default length

        for idx, p in enumerate(players_in_round):
            if not p or not hasattr(p, 'id'): continue # Skip invalid player entries

            is_acting = (p.id == player_id_acting)
            if is_acting:
                acting_player_obj = p

            # Player Name and Money
            name_str = (p.name if hasattr(p, 'name') else f'P{p.id}').ljust(max_name_len)
            money_val = p.money if hasattr(p, 'money') else 0
            money_str = f"${money_val}".ljust(6)

            # Player Cards (Show only for the acting player)
            cards_str = "( ? ? )" # Default hidden cards
            if is_acting and p.id in table_state._player_cards:
                cards_str = f"( {format_cards_for_ai(table_state._player_cards[p.id])} )"

            # Player Status Flags
            status = []
            if hasattr(p, 'is_folded') and p.is_folded: status.append("FOLDED")
            if hasattr(p, 'is_all_in') and p.is_all_in: status.append("ALL-IN")
            # Dealer/Blind Indicators (based on index in the current round's player list)
            if hasattr(table_state.round, 'button') and idx == table_state.round.button: status.append("D")
            if hasattr(table_state.round, 'small_blind_player_index') and idx == table_state.round.small_blind_player_index: status.append("SB")
            if hasattr(table_state.round, 'big_blind_player_index') and idx == table_state.round.big_blind_player_index: status.append("BB")
            status_str = f"[{' '.join(status)}]" if status else ""

            # Current Bet in this Round Stage
            current_bet = 0
            if (hasattr(p, 'turn_stake') and isinstance(p.turn_stake, list) and
                hasattr(table_state.round.turn, 'value') and
                len(p.turn_stake) > table_state.round.turn.value):
                current_bet = p.turn_stake[table_state.round.turn.value]
            bet_str = f"Bet:${current_bet}" if current_bet > 0 else ""

            # Identify the acting player
            prefix = "**YOU ->**" if is_acting else "   -"
            lines.append(f"{prefix} {name_str} {money_str} {cards_str.ljust(10)} {bet_str.ljust(10)} {status_str}")

        # --- Action Context for the Acting Player ---
        lines.append("\n**Action Context:**")
        current_round_bet = 0 # Bet level for the current street
        if table_state.round and hasattr(table_state.round, 'turn_stake'):
            current_round_bet = table_state.round.turn_stake
        lines.append(f"- Current Bet to Match: ${current_round_bet}")

        amount_to_call = table_state.to_call # Amount needed for THIS player to call
        lines.append(f"- Amount You Need to Call: ${amount_to_call}")
        acting_player_money = 'N/A'
        if acting_player_obj and hasattr(acting_player_obj, 'money'):
            acting_player_money = acting_player_obj.money
        lines.append(f"- Your Stack: ${acting_player_money}")

        # Possible Actions (Based on flags set in publicOut handler)
        possible_actions = ["FOLD"]
        if table_state.can_check:
            possible_actions.append("CHECK")
        elif amount_to_call > 0 and (acting_player_obj and acting_player_money > 0):
            call_amount_str = min(amount_to_call, acting_player_money) # Effective call amount
            possible_actions.append(f"CALL({call_amount_str})")
        if table_state.can_raise:
            possible_actions.append("RAISE")
        lines.append(f"- Possible Actions: {', '.join(possible_actions)}")

        # --- Task Prompt ---
        lines.append(f"\n**Task:** Respond ONLY with the action name: {', '.join(['CALL' if 'CALL(' in a else a for a in possible_actions])}.")
        lines.append("If RAISING, add ' AMOUNT: X' on the same line, where X is the integer amount to raise BY (the additional amount on top of any call). Example: 'RAISE AMOUNT: 20'")

        return "\n".join(lines)

    except Exception as e:
        # Catch-all for unexpected errors during formatting
        ai_logger.error(f"Error during format_state_for_ai: {e}", exc_info=True)
        return f"Error: Could not format game state - {e}"

# ==============================================================================
# Custom Table Class
# ==============================================================================

class CommandLineTable(Table):
    """
    Custom Table class extending pokerlib's Table to handle command-line interaction,
    AI player management, game state display, probability calculation, and specific event handling.
    """
    def __init__(self, *args, **kwargs):
        """Initializes the CommandLineTable."""
        super().__init__(*args, **kwargs)
        # --- Game State Tracking ---
        self._player_cards = {}             # Stores hole cards: {player_id: ((Rank, Suit), (Rank, Suit)), ...}
        self._current_action_player_id = None # ID of the player whose action is currently required.
        self.round_over_flag = False        # Set to True when RoundPublicOutId.ROUNDFINISHED is received.
        self.to_call = 0                    # Amount the current player needs to call.
        self.can_check = False              # If the current player can check.
        self.can_raise = False              # If the current player can raise.
        self.min_raise = 0                  # Minimum raise amount (usually big blind).
        self.last_turn_processed = -1       # Tracks the last turn index where probabilities were calculated

        # --- AI Management ---
        # Stores message history per round per AI player: {round_id: {player_id: [messages]}}
        self.ai_message_history = defaultdict(lambda: defaultdict(list))
        # Stores AI model assignment: {player_id: 'model_name'}
        self.ai_model_assignments = {}

        # --- Deuces Integration ---
        self.evaluator = Evaluator()        # Deuces evaluator instance
        self._player_probabilities = {}     # Stores win probabilities: {player_id: float_percentage}
        self._deuces_full_deck = Deck.GetFullDeck() # Keep a reference to the full deck for conversion

        # --- End-of-Round Display Buffers ---
        self.pending_winner_messages = []
        self.pending_showdown_messages = []
        self.printed_showdown_board = False
        self.showdown_occurred = False

    def _convert_pokerlib_to_deuces(self, card_tuple):
        """Converts a pokerlib card tuple (Rank, Suit) to its Deuces integer representation."""
        if not card_tuple or len(card_tuple) != 2:
            return None
        rank, suit = card_tuple
        rank_str = RANK_MAP_TO_DEUCES.get(rank)
        suit_str = SUIT_MAP_TO_DEUCES.get(suit)
        if rank_str is None or suit_str is None:
            ai_logger.warning(f"Failed to convert pokerlib card to deuces: Rank={rank}, Suit={suit}")
            return None
        try:
            # Card.new() creates the integer representation from the string
            return Card.new(rank_str + suit_str)
        except Exception as e:
            ai_logger.error(f"Error creating Deuces card for {rank_str}{suit_str}: {e}")
            return None

    def _calculate_win_probabilities(self):
        """
        Calculates win probabilities for all active players using Monte Carlo simulation
        with the Deuces library. Stores results in self._player_probabilities.
        """
        if not SHOW_PROBABILITIES:
            self._player_probabilities.clear() # Ensure it's clear if disabled
            return
        if not self.round or not hasattr(self.round, 'players'):
            ai_logger.warning("Cannot calculate probabilities: Invalid round or no players.")
            return

        start_time = time.time()
        # Use a status variable to check if probabilities were calculated
        prob_calculated = False
        current_stage = self.round.turn.name # Get current stage name

        print(colorize(f"\n[Calculating Probabilities for {current_stage} - {PROBABILITY_SIMULATIONS} sims...]", Colors.BRIGHT_BLACK))
        ai_logger.info(f"Starting probability calculation for {current_stage}...")

        # 1. Identify active players and their known hands
        active_players = {} # { player_id: [deuces_card1, deuces_card2] }
        known_cards_deuces = [] # List to store all known card integers (board + player hands)
        board_cards_pokerlib = self.round.board if hasattr(self.round, 'board') else []
        board_cards_deuces = [self._convert_pokerlib_to_deuces(c) for c in board_cards_pokerlib]
        board_cards_deuces = [c for c in board_cards_deuces if c is not None] # Filter out conversion errors
        known_cards_deuces.extend(board_cards_deuces)

        players_in_round_pokerlib = self.round.players
        active_player_ids = [] # Keep track of IDs for the simulation loop

        for p in players_in_round_pokerlib:
            if p and hasattr(p, 'id') and hasattr(p, 'is_folded') and not p.is_folded:
                player_id = p.id
                if player_id in self._player_cards:
                    hand_pokerlib = self._player_cards[player_id]
                    hand_deuces = [self._convert_pokerlib_to_deuces(c) for c in hand_pokerlib]
                    if len(hand_deuces) == 2 and None not in hand_deuces:
                        active_players[player_id] = hand_deuces
                        known_cards_deuces.extend(hand_deuces)
                        active_player_ids.append(player_id)
                    else:
                        ai_logger.warning(f"Player {player_id} active but failed to convert hand {hand_pokerlib} to deuces.")
                else:
                    ai_logger.warning(f"Player {player_id} active but no hand found in _player_cards.")

        num_active_players = len(active_players)
        if num_active_players < 2:
            ai_logger.info("Skipping probability calculation: Less than 2 active players.")
            # Set probabilities to 100% for the single player or 0% if none
            self._player_probabilities.clear()
            for pid in active_player_ids: self._player_probabilities[pid] = 100.0
            return

        # 2. Create the deck of unknown cards for simulation
        # Start with a full deck and remove known cards
        sim_deck_integers = list(self._deuces_full_deck) # Copy the full deck list
        for card_int in known_cards_deuces:
            try:
                sim_deck_integers.remove(card_int)
            except ValueError:
                # This *shouldn't* happen if known_cards are valid, but log if it does
                ai_logger.error(f"Known card {Card.int_to_str(card_int)} not found in simulation deck for removal.")
                # Consider halting calculation if deck state is compromised
                # return

        # Check if deck size makes sense
        expected_deck_size = 52 - len(known_cards_deuces)
        if len(sim_deck_integers) != expected_deck_size:
             ai_logger.error(f"Simulation deck size mismatch! Expected {expected_deck_size}, got {len(sim_deck_integers)}. Known: {len(known_cards_deuces)}. Aborting calc.")
             print(colorize("[ERROR] Probability calculation failed due to deck inconsistency.", Colors.RED))
             return


        # 3. Run Monte Carlo Simulations
        win_counts = {player_id: 0 for player_id in active_player_ids}
        tie_counts = {player_id: 0 for player_id in active_player_ids} # Optional: track ties separately
        num_board_cards_needed = 5 - len(board_cards_deuces)

        for i in range(PROBABILITY_SIMULATIONS):
            # Create a fresh *copy* of the unknown cards deck for this simulation run
            current_sim_deck = list(sim_deck_integers)
            random.shuffle(current_sim_deck)
            draw_pointer = 0

            # Deal remaining board cards
            sim_board_extension = []
            if num_board_cards_needed > 0:
                 sim_board_extension = current_sim_deck[draw_pointer : draw_pointer + num_board_cards_needed]
                 draw_pointer += num_board_cards_needed
            final_sim_board = board_cards_deuces + sim_board_extension

            # Deal hands to opponents (if any) - Each active player needs 2 cards
            player_sim_hands = {}
            possible_draw_error = False
            for pid in active_player_ids:
                # Player already has their known hand from active_players dict
                if draw_pointer + 2 <= len(current_sim_deck):
                    # Only need to assign the known hand for evaluation later
                    player_sim_hands[pid] = active_players[pid]
                    # We don't actually draw cards from the deck *for them* here,
                    # their cards are already known and removed from sim_deck_integers.
                    # We *do* need to draw cards for *opponents* in a heads-up equity calc,
                    # but here we evaluate all active players against each other.
                else:
                    # This indicates an error in deck management or logic
                    ai_logger.error(f"Simulation Error: Not enough cards left in deck ({len(current_sim_deck)}) to deal hand during sim {i+1}. Pointer: {draw_pointer}")
                    possible_draw_error = True
                    break # Stop this simulation

            if possible_draw_error:
                continue # Skip to next simulation iteration

            # Evaluate hands for all active players
            scores = {}
            for pid in active_player_ids:
                 # Deuces evaluates the best 5-card hand from 7 (2 hole + 5 board)
                 # or fewer if board isn't complete yet (but simulation completes it)
                 hand_to_eval = player_sim_hands[pid] + final_sim_board
                 # Ensure we have exactly 5, 6, or 7 cards for deuces evaluate
                 if len(hand_to_eval) >= 5:
                    scores[pid] = self.evaluator.evaluate(player_sim_hands[pid], final_sim_board)
                 else:
                    # Should not happen if simulation deals correctly
                    ai_logger.warning(f"Sim {i+1}: Incorrect number of cards ({len(hand_to_eval)}) for evaluation for player {pid}. Hand: {player_sim_hands[pid]}, Board: {final_sim_board}")
                    scores[pid] = 9999 # Assign a very bad score

            # Determine winner(s) for this simulation (lower score is better)
            if not scores: continue # Skip if no scores were calculated (e.g., due to errors)

            best_score = min(scores.values())
            winners = [pid for pid, score in scores.items() if score == best_score]

            if len(winners) == 1:
                win_counts[winners[0]] += 1
            else:
                # Handle ties - distribute the "win" among tied players
                for winner_id in winners:
                    win_counts[winner_id] += 1.0 / len(winners) # Fractional win for ties
                    tie_counts[winner_id] += 1 # Optional: Count explicit ties


        # 4. Calculate and Store Percentages
        self._player_probabilities.clear() # Clear previous results
        for pid in active_player_ids:
            # Using fractional wins for ties directly gives equity
            win_percentage = (win_counts[pid] / PROBABILITY_SIMULATIONS) * 100.0
            self._player_probabilities[pid] = win_percentage
            prob_calculated = True # Mark that we calculated something

        end_time = time.time()
        duration = end_time - start_time
        if prob_calculated:
           print(colorize(f"[Probability calculation complete ({duration:.2f}s)]", Colors.BRIGHT_BLACK))
           ai_logger.info(f"Probability calculation for {current_stage} finished in {duration:.2f}s. Results: {self._player_probabilities}")
        else:
           print(colorize("[Probability calculation skipped or failed.]", Colors.YELLOW))
           ai_logger.warning(f"Probability calculation for {current_stage} did not produce results.")


    def assign_ai_models(self):
        """Assigns AI models from AI_MODEL_LIST cyclically to AI players at the table."""
        ai_player_ids = []
        player_group = self.seats.getPlayerGroup()
        if not player_group: return # No players to assign to

        # Identify AI player IDs
        for p in player_group:
             if p and hasattr(p, 'id') and (AI_ONLY_MODE or p.id != HUMAN_PLAYER_ID):
                 ai_player_ids.append(p.id)

        num_ai = len(ai_player_ids)
        if num_ai == 0 or not AI_MODEL_LIST:
            ai_logger.info("No AI players or no AI models defined. Skipping model assignment.")
            return # No AI players or no models defined

        log_msg_assign = ["AI Model Assignments:"]
        self.ai_model_assignments.clear() # Clear previous assignments

        # Assign models cyclically
        for i, pid in enumerate(ai_player_ids):
            model_full_name = AI_MODEL_LIST[i % len(AI_MODEL_LIST)] # Cycle through models
            self.ai_model_assignments[pid] = model_full_name        # Store the full name

            # Get short name for logging/display
            model_short_name = AI_MODEL_SHORT_NAMES.get(model_full_name, model_full_name.split('/')[-1])
            player_name = self._get_player_name(pid)
            assign_text = f"AI Player {player_name} assigned model: {model_short_name} ({model_full_name})"
            # Optional: print assignment to console
            # print(colorize(assign_text, Colors.MAGENTA))
            log_msg_assign.append(f"  - Player {pid} ({player_name}): {model_full_name} (as {model_short_name})")

        # Log the assignments
        if len(log_msg_assign) > 1:
            ai_logger.info("\n".join(log_msg_assign))

    def _get_player_name(self, player_id):
        """Safely retrieves a player's name by their ID."""
        player = self.seats.getPlayerById(player_id)
        return player.name if player and hasattr(player, 'name') else f"P{player_id}"

    def _newRound(self, round_id):
        """Handles the initialization of a new round.

        Resets round-specific state, assigns AI models if necessary,
        and initializes the Round object. Also clears probabilities.

        Args:
            round_id: The identifier for the new round.
        """
        ai_logger.info(f"\n{'='*15} ROUND {round_id} START {'='*15}")
        current_players = self.seats.getPlayerGroup()

        if not current_players:
            print(colorize("Error: Cannot start round - No players at the table.", Colors.RED))
            ai_logger.error(f"Attempted to start round {round_id} with no players.")
            self.round = None # Ensure round object is None
            return

        # Reset player states (e.g., folded status) if method exists
        for player in current_players:
            if hasattr(player, "resetState"):
                 try: player.resetState()
                 except Exception as e: ai_logger.warning(f"Could not reset state for P{player.id}: {e}")

        # --- Reset Round-Specific State ---
        self._player_cards.clear()           # Clear stored hole cards from previous round
        self.pending_winner_messages.clear()
        self.pending_showdown_messages.clear()
        self.printed_showdown_board = False
        self.showdown_occurred = False
        self.round_over_flag = False        # IMPORTANT: Reset round over flag here
        self._current_action_player_id = None # Clear any lingering action request
        self.to_call = 0
        self.can_check = False
        self.can_raise = False
        self.min_raise = self.big_blind      # Reset min raise to default
        self._player_probabilities.clear()   # <<< Clear probabilities for the new round >>>
        self.last_turn_processed = -1        # Reset turn tracking for probability calc

        # Clear AI message history for the new round ID if it exists (optional)
        if round_id in self.ai_message_history:
            self.ai_message_history[round_id].clear()

        # Assign AI models if needed (e.g., if players changed or first round)
        num_ai_needed = len([p for p in current_players if AI_ONLY_MODE or (p and hasattr(p, 'id') and p.id != HUMAN_PLAYER_ID)])
        if num_ai_needed > 0 and not self.ai_model_assignments :
            self.assign_ai_models()

        # --- Initialize the pokerlib Round ---
        try:
            self.round = self.RoundClass(
                round_id,
                current_players,
                self.button,
                self.small_blind,
                self.big_blind
            )
            ai_logger.info(f"Successfully initialized pokerlib Round {round_id}.")
            # Note: Cards are dealt *after* this via privateOut,
            # so pre-flop probabilities must be calculated later.
        except Exception as e:
            print(colorize(f"--- ROUND INITIALIZATION ERROR (Round {round_id}) ---", Colors.RED))
            print(traceback.format_exc()) # Print detailed error
            ai_logger.error(f"Failed to initialize pokerlib Round {round_id}: {e}", exc_info=True)
            self.round = None # Ensure round object is None if init fails

    def _display_game_state(self, clear_override=False):
        """Displays the current game state to the terminal, including win probabilities if enabled.

        Args:
            clear_override: If True, bypasses the CLEAR_SCREEN check and does NOT clear.
        """
        # Clear terminal conditionally
        if not clear_override and CLEAR_SCREEN:
            clear_terminal()

        # --- Header ---
        title = colorize("====== POKER GAME STATE ======", Colors.BRIGHT_CYAN + Colors.BOLD)
        separator = colorize("--------------------------------------------------", Colors.BRIGHT_BLACK)
        print(f"\n{title}")

        # --- Check if Round Data is Valid ---
        round_valid = ( self.round and hasattr(self.round, 'id') and
                        hasattr(self.round, 'players') and isinstance(self.round.players, list) and
                        hasattr(self.round, 'turn') and hasattr(self.round, 'board') and
                        hasattr(self.round, 'pot_size') and isinstance(self.round.pot_size, list) )

        if not round_valid:
            # Display minimal info if round data is missing or invalid
            print(colorize("No active round or state data unavailable.", Colors.YELLOW))
            print(colorize("\nPlayers at table:", Colors.YELLOW))
            for player in self.seats.getPlayerGroup():
                if not player or not hasattr(player, 'id'): continue
                money_str = f"${player.money}" if hasattr(player, 'money') else colorize("N/A", Colors.BRIGHT_BLACK)
                is_ai = AI_ONLY_MODE or player.id != HUMAN_PLAYER_ID
                display_name = player.name if hasattr(player, 'name') else f"P{player.id}"
                type_indicator = colorize(" (AI)", Colors.MAGENTA) if is_ai else colorize(" (Human)", Colors.GREEN)
                print(f"  - {colorize(display_name, Colors.CYAN)}{type_indicator}: {colorize(money_str, Colors.BRIGHT_GREEN)}")
            print(separator)
            return

        # --- Display Round Details (Safely) ---
        try:
            round_id = self.round.id
            turn_name = self.round.turn.name
            board_cards_list = [tuple(c) for c in self.round.board if isinstance(c, (list,tuple)) and len(c)==2]
            board_cards_str = format_cards_terminal(board_cards_list)
            pot_total = sum(self.round.pot_size)
            current_turn_enum = self.round.turn
            players_in_round = self.round.players
            button_idx = self.round.button if hasattr(self.round, "button") else -1
            sb_idx = self.round.small_blind_player_index if hasattr(self.round, "small_blind_player_index") else -1
            bb_idx = self.round.big_blind_player_index if hasattr(self.round, "big_blind_player_index") else -1

            print(f"Round: {colorize(str(round_id), Colors.WHITE)}   Turn: {colorize(turn_name, Colors.WHITE + Colors.BOLD)}")
            print(f"Board: [ {board_cards_str} ]")
            print(f"Pot:   {colorize(f'${pot_total}', Colors.BRIGHT_YELLOW + Colors.BOLD)}")
            print(colorize("\nPlayers:", Colors.YELLOW))

        except (AttributeError, TypeError, IndexError) as e:
            print(colorize(f"Error accessing round details: {e}", Colors.RED))
            ai_logger.warning(f"Error accessing round details in _display_game_state: {e}")
            print(separator)
            return # Stop display if essential round info is bad

        # --- Display Player Details ---
        max_name_len = 0
        if players_in_round:
             try:
                 max_name_len = max(len(p.name if hasattr(p,'name') else f'P{p.id}') for p in players_in_round if p and hasattr(p, 'id'))
             except ValueError:
                 max_name_len = 10

        # Define fixed widths for layout
        name_width = max_name_len + 2
        money_width = 8
        cards_width = 12 # Width for "( XX XX )"
        prob_width = 10  # Width for " (XX.X%)"
        stake_width = 12 # Width for "[Bet: $XXX]"

        for idx, player in enumerate(players_in_round):
            if not player or not hasattr(player, "id"): continue

            player_id = player.id
            is_acting = (player_id == self._current_action_player_id)
            line_prefix = colorize(" > ", Colors.BRIGHT_YELLOW + Colors.BOLD) if is_acting else "   "

            # Player Name and Money
            player_name_str = (player.name if hasattr(player, 'name') else f'P{player_id}')
            player_name_colored = colorize(player_name_str.ljust(name_width), Colors.CYAN + (Colors.BOLD if is_acting else ""))
            money_val = player.money if hasattr(player, 'money') else 0
            money_str = colorize(f"${money_val}", Colors.BRIGHT_GREEN).ljust(money_width)

            # --- Card Display Logic ---
            cards_str = colorize("( ? ? )", Colors.BRIGHT_BLACK) # Default hidden
            show_cards = False
            player_folded = hasattr(player, 'is_folded') and player.is_folded

            if player_id in self._player_cards:
                # Determine if cards should be shown
                if not AI_ONLY_MODE and player_id == HUMAN_PLAYER_ID and not player_folded:
                    show_cards = True
                elif AI_ONLY_MODE and not player_folded: # Show all non-folded AI cards in AI Only mode
                    show_cards = True
                elif self.round_over_flag and not player_folded and self.showdown_occurred: # Show cards at showdown
                    show_cards = True

            if show_cards and player_id in self._player_cards:
                 cards_str = f"( {format_cards_terminal(self._player_cards[player_id])} )"
            elif player_folded:
                 cards_str = colorize("(FOLDED)", Colors.BRIGHT_BLACK)

            cards_formatted = cards_str.ljust(cards_width) # Apply padding

            # --- Probability Display ---
            prob_str = ""
            if SHOW_PROBABILITIES and not player_folded and player_id in self._player_probabilities:
                prob = self._player_probabilities[player_id]
                prob_str = colorize(f"({prob:.1f}%)", Colors.BRIGHT_BLUE) # Display with one decimal place

            prob_formatted = prob_str.ljust(prob_width) # Apply padding

            # --- Status and Bet ---
            status = []
            if hasattr(player, 'is_all_in') and player.is_all_in: status.append(colorize("ALL-IN", Colors.BRIGHT_RED + Colors.BOLD))
            if idx == button_idx: status.append(colorize("D", Colors.WHITE + Colors.BOLD))
            if idx == sb_idx: status.append(colorize("SB", Colors.YELLOW))
            if idx == bb_idx: status.append(colorize("BB", Colors.YELLOW))
            status_str = " ".join(status)

            turn_stake_val = 0
            if ( hasattr(player, "turn_stake") and isinstance(player.turn_stake, list) and
                 hasattr(current_turn_enum, 'value') and
                 len(player.turn_stake) > current_turn_enum.value ):
                turn_stake_val = player.turn_stake[current_turn_enum.value]
            stake_str = colorize(f"[Bet: ${turn_stake_val}]", Colors.MAGENTA) if turn_stake_val > 0 else ""
            stake_formatted = stake_str.ljust(stake_width) # Apply padding

            # --- Assemble and Print Line ---
            # Order: Prefix Name Money Cards Probability Bet Status
            print(f"{line_prefix}{player_name_colored}{money_str}{cards_formatted}{prob_formatted}{stake_formatted} {status_str}")

        print(separator)

    def publicOut(self, out_id, **kwargs):
        """Handles public events broadcast by the poker engine.

        Prints game flow info, triggers probability calculations, and manages display updates.
        """
        player_id = kwargs.get("player_id")
        player_name_raw = self._get_player_name(player_id) if player_id else "System"
        player_name = colorize(player_name_raw, Colors.CYAN)
        msg = ""
        prefix = ""
        processed = False
        recalc_probs = False # Flag to trigger probability recalc after handling event
        current_turn_idx = self.round.turn.value if self.round and hasattr(self.round, 'turn') else -1

        is_round_event = isinstance(out_id, RoundPublicOutId)
        is_table_event = isinstance(out_id, TablePublicOutId)

        player_action_events = {
            RoundPublicOutId.PLAYERCHECK, RoundPublicOutId.PLAYERCALL,
            RoundPublicOutId.PLAYERFOLD, RoundPublicOutId.PLAYERRAISE,
            RoundPublicOutId.PLAYERWENTALLIN
        }

        # --- Round Events ---
        if is_round_event:
            processed = True
            prefix = colorize("[ROUND]", Colors.BLUE)
            player = self.seats.getPlayerById(player_id) if player_id else None

            if out_id == RoundPublicOutId.NEWROUND:
                prefix = colorize("[ROUND]", Colors.BLUE)
                msg = f"Dealing cards for Round {kwargs.get('round_id', '?')}..."
                ai_logger.info(f"Received NEWROUND event (ID: {kwargs.get('round_id', '?')})")
                # Probabilities calculated *after* cards are dealt (in PLAYERACTIONREQUIRED handler for preflop)
            elif out_id == RoundPublicOutId.NEWTURN:
                prefix = ""
                msg = ""
                turn_enum = kwargs.get('turn')
                ai_logger.info(f"Received NEWTURN event: {turn_enum}")
                if turn_enum and turn_enum != Turn.PREFLOP: # Recalculate for Flop, Turn, River
                    if current_turn_idx != self.last_turn_processed: # Avoid recalc if turn didn't advance
                         recalc_probs = True
                         self.last_turn_processed = current_turn_idx # Mark this turn as processed
                if not self.round_over_flag:
                    # Display state *after* potential recalc for Flop/Turn/River
                    pass # Display will happen after recalc or in action required
            elif out_id == RoundPublicOutId.SMALLBLIND:
                msg = f"{player_name} posts {colorize('SB', Colors.YELLOW)} ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.BIGBLIND:
                msg = f"{player_name} posts {colorize('BB', Colors.YELLOW)} ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.PLAYERCHECK:
                prefix = colorize("[ACTION]", Colors.GREEN)
                msg = f"{player_name} checks"
            elif out_id == RoundPublicOutId.PLAYERCALL:
                prefix = colorize("[ACTION]", Colors.GREEN)
                msg = f"{player_name} calls ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.PLAYERFOLD:
                prefix = colorize("[ACTION]", Colors.BRIGHT_BLACK)
                msg = f"{player_name} folds"
                # Optional: Recalc probabilities after fold if > 2 players remain? Can be slow.
                # active_count = sum(1 for p in self.round.players if hasattr(p,'is_folded') and not p.is_folded)
                # if active_count >= 2: recalc_probs = True
            elif out_id == RoundPublicOutId.PLAYERRAISE:
                prefix = colorize("[ACTION]", Colors.BRIGHT_MAGENTA)
                msg = f"{player_name} raises by ${kwargs['raised_by']} (total bet this street: ${kwargs['paid_amount']})"
            elif out_id == RoundPublicOutId.PLAYERISALLIN:
                prefix = colorize("[INFO]", Colors.BRIGHT_RED)
                msg = f"{player_name} is {colorize('ALL-IN!', Colors.BOLD)}"
            elif out_id == RoundPublicOutId.PLAYERWENTALLIN:
                prefix = colorize("[ACTION]", Colors.BRIGHT_RED + Colors.BOLD)
                msg = f"{player_name} goes ALL-IN with ${kwargs['paid_amount']}!"

            elif out_id == RoundPublicOutId.PLAYERACTIONREQUIRED:
                 prefix = ""
                 msg = ""
                 self._current_action_player_id = player_id
                 self.to_call = kwargs.get('to_call', 0)
                 self.can_check = self.to_call == 0
                 self.can_raise = player and hasattr(player, 'money') and player.money > self.to_call
                 self.min_raise = self.big_blind
                 ai_logger.info(f"Action required for P{player_id} ({player_name_raw}). To Call: {self.to_call}, Can Check: {self.can_check}, Can Raise: {self.can_raise}")

                 # <<< Trigger Pre-flop Probability Calculation >>>
                 # Calculate only if it's PREFLOP and probabilities haven't been calculated yet for this round.
                 if self.round.turn == Turn.PREFLOP and current_turn_idx > self.last_turn_processed:
                     recalc_probs = True
                     self.last_turn_processed = current_turn_idx # Mark preflop as processed


            elif out_id == RoundPublicOutId.PUBLICCARDSHOW:
                prefix = colorize("[SHOWDOWN]", Colors.WHITE)
                msg = "" # Buffered message

                if not self.printed_showdown_board and self.round and hasattr(self.round, 'board'):
                    board_cards_list = [tuple(c) for c in self.round.board if isinstance(c, (list,tuple)) and len(c)==2]
                    print()
                    print(f"{colorize('[BOARD]', Colors.WHITE)} {format_cards_terminal(board_cards_list)}")
                    self.printed_showdown_board = True
                    ai_logger.info(f"Showdown: Board printed: {format_cards_for_ai(board_cards_list)}")

                shown_cards_raw = kwargs.get('cards', [])
                shown_cards_tuples = [tuple(c) for c in shown_cards_raw if isinstance(c, (list, tuple)) and len(c)==2]

                if player and shown_cards_tuples:
                    self._player_cards[player.id] = tuple(shown_cards_tuples)
                    showdown_msg = f"{prefix} {player_name} shows {format_cards_terminal(shown_cards_tuples)}"
                    if showdown_msg not in self.pending_showdown_messages:
                        self.pending_showdown_messages.append(showdown_msg)
                        ai_logger.info(f"Buffered Showdown: {player_name_raw} shows {format_cards_for_ai(shown_cards_tuples)}")
                    self.showdown_occurred = True
                else:
                    ai_logger.warning(f"Received PUBLICCARDSHOW for {player_name_raw} but invalid player or cards: {shown_cards_raw}")

            elif out_id == RoundPublicOutId.DECLAREPREMATUREWINNER:
                prefix = colorize("[WINNER]", Colors.BRIGHT_YELLOW + Colors.BOLD)
                winner_msg = f"{prefix} {player_name} wins ${kwargs['money_won']} (Prematurely - all others folded)"
                if winner_msg not in self.pending_winner_messages:
                    self.pending_winner_messages.append(winner_msg)
                    ai_logger.info(f"Buffered Premature Winner: {player_name_raw} wins ${kwargs['money_won']}")
                msg = ""
            elif out_id == RoundPublicOutId.DECLAREFINISHEDWINNER:
                prefix = colorize("[WINNER]", Colors.BRIGHT_YELLOW + Colors.BOLD)
                hand_name = format_hand_enum(kwargs.get('handname'))
                winner_msg = f"{prefix} {player_name} wins ${kwargs['money_won']} with {hand_name}"
                if winner_msg not in self.pending_winner_messages:
                    self.pending_winner_messages.append(winner_msg)
                    ai_logger.info(f"Buffered Finished Winner: {player_name_raw} wins ${kwargs['money_won']} with {hand_name}")
                msg = ""

            elif out_id == RoundPublicOutId.ROUNDFINISHED:
                prefix = ""
                msg = ""
                self.round_over_flag = True
                ai_logger.info(f"Received ROUNDFINISHED event.")

                # Print buffered info
                if self.showdown_occurred and not self.printed_showdown_board and self.round and hasattr(self.round, 'board'):
                     board_cards_list = [tuple(c) for c in self.round.board if isinstance(c, (list,tuple)) and len(c)==2]
                     print()
                     print(f"{colorize('[BOARD]', Colors.WHITE)} {format_cards_terminal(board_cards_list)}")
                     self.printed_showdown_board = True
                     ai_logger.info(f"Showdown: Board printed at ROUNDFINISHED: {format_cards_for_ai(board_cards_list)}")

                for showdown_msg in self.pending_showdown_messages: print(showdown_msg)
                for winner_msg in self.pending_winner_messages: print(winner_msg)

        # --- Table Events ---
        elif is_table_event:
            processed = True
            prefix = colorize("[TABLE]", Colors.MAGENTA)
            if out_id == TablePublicOutId.PLAYERJOINED:
                msg = f"{player_name} joined seat {kwargs['player_seat']}"
            elif out_id == TablePublicOutId.PLAYERREMOVED:
                 msg = f"{player_name} left the table."
                 if player_id in self._player_cards: del self._player_cards[player_id]
                 if player_id in self.ai_model_assignments: del self.ai_model_assignments[player_id]
                 if player_id in self._player_probabilities: del self._player_probabilities[player_id]
                 if player_id in self.ai_message_history:
                     for round_hist in self.ai_message_history.values():
                         if player_id in round_hist: del round_hist[player_id]
            elif out_id == TablePublicOutId.NEWROUNDSTARTED:
                prefix = ""; msg = ""
            elif out_id == TablePublicOutId.ROUNDNOTINITIALIZED:
                prefix = colorize("[ERROR]", Colors.RED); msg = "No round is currently running."
            elif out_id == TablePublicOutId.ROUNDINPROGRESS:
                prefix = colorize("[ERROR]", Colors.RED); msg = "A round is already in progress."
            elif out_id == TablePublicOutId.INCORRECTNUMBEROFPLAYERS:
                prefix = colorize("[ERROR]", Colors.RED); msg = "Incorrect number of players to start (need 2+)."

        # --- Final Print Decision & Logging ---
        should_print = bool(msg)
        if is_round_event and out_id in [
            RoundPublicOutId.NEWTURN, RoundPublicOutId.ROUNDFINISHED,
            RoundPublicOutId.PUBLICCARDSHOW, RoundPublicOutId.DECLAREFINISHEDWINNER,
            RoundPublicOutId.DECLAREPREMATUREWINNER, RoundPublicOutId.PLAYERACTIONREQUIRED,
            ]:
             should_print = False
        if is_table_event and out_id == TablePublicOutId.NEWROUNDSTARTED:
             should_print = False

        if out_id not in [RoundPublicOutId.NEWTURN, RoundPublicOutId.PLAYERACTIONREQUIRED]:
            log_level = logging.WARNING if "ERROR" in prefix else logging.INFO
            ai_logger.log(log_level, f"PublicOut Event: ID={out_id}, Args={kwargs}, Message='{msg}' (Printed: {should_print})")

        if should_print:
            print(f"{prefix} {msg}")
            if is_round_event and out_id in player_action_events:
                is_ai_player = AI_ONLY_MODE or (player_id is not None and player_id != HUMAN_PLAYER_ID)
                if is_ai_player:
                    time.sleep(1.5) # Pause after AI action message

        elif not processed:
             unhandled_msg = f"Unhandled PublicOut Event: ID={out_id} Data: {kwargs}"
             print(colorize(unhandled_msg, Colors.BRIGHT_BLACK))
             ai_logger.warning(unhandled_msg)

        # --- Recalculate Probabilities and Refresh Display ---
        if recalc_probs and SHOW_PROBABILITIES:
            self._calculate_win_probabilities()
            # Refresh display after calculation if it's not the end of the round yet
            if not self.round_over_flag:
                 self._display_game_state() # Show updated state with new probabilities

        # If an action required event occurred, redisplay the state (will show probs if calculated)
        if is_round_event and out_id == RoundPublicOutId.PLAYERACTIONREQUIRED and not self.round_over_flag:
            self._display_game_state()


    def privateOut(self, player_id, out_id, **kwargs):
        """Handles private events sent to a specific player."""
        player_name_raw = self._get_player_name(player_id)
        player_name_color = colorize(player_name_raw, Colors.CYAN)
        prefix = colorize(f"[PRIVATE to {player_name_raw}]", Colors.YELLOW)
        msg = ""
        log_msg = f"PrivateOut for P{player_id} ({player_name_raw}): ID={out_id}, Args={kwargs}"
        log_level = logging.INFO

        if isinstance(out_id, RoundPrivateOutId):
            if out_id == RoundPrivateOutId.DEALTCARDS:
                cards_raw = kwargs.get('cards')
                if cards_raw and isinstance(cards_raw, (list, tuple)) and len(cards_raw) == 2:
                    cards_tuples = tuple(tuple(c) for c in cards_raw if isinstance(c, (list, tuple)) and len(c)==2)
                    if len(cards_tuples) == 2:
                        self._player_cards[player_id] = cards_tuples
                        card_str_terminal = format_cards_terminal(cards_tuples)
                        card_str_log = format_cards_for_ai(cards_tuples)
                        msg = f"You are dealt {card_str_terminal}"
                        log_msg += f" - Cards: {card_str_log}"
                        # Note: Pre-flop probability calculation is triggered later in publicOut
                    else:
                        msg = colorize("Card data conversion error.", Colors.RED)
                        log_msg += " - Error: Could not convert raw cards to tuples."
                        log_level = logging.ERROR
                else:
                    msg = colorize("Card dealing error (invalid data received).", Colors.RED)
                    log_msg += f" - Error: Invalid card data received: {cards_raw}"
                    log_level = logging.ERROR

        elif isinstance(out_id, TablePrivateOutId):
            prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED)
            log_level = logging.ERROR
            if out_id == TablePrivateOutId.BUYINTOOLOW:
                 msg = f"Your buy-in is too low. Minimum required: ${self.buyin}."
            elif out_id == TablePrivateOutId.TABLEFULL:
                 msg = f"The table is currently full."
            elif out_id == TablePrivateOutId.PLAYERALREADYATTABLE:
                 msg = f"You are already seated at this table."
            elif out_id == TablePrivateOutId.PLAYERNOTATTABLE:
                 msg = f"You are not seated at this table."
            elif out_id == TablePrivateOutId.INCORRECTSEATINDEX:
                 msg = f"Invalid seat index specified."
            else:
                 prefix = colorize(f"[UNHANDLED PRIVATE to {player_name_raw}]", Colors.BRIGHT_BLACK)
                 msg = f"Unknown TablePrivateOutId: {out_id} Data: {kwargs}"
                 log_level = logging.WARNING

        # --- Logging ---
        ai_logger.log(log_level, log_msg)

        # --- Print to Console (Only for Human Player in Mixed Mode) ---
        if msg and not AI_ONLY_MODE and player_id == HUMAN_PLAYER_ID:
            print(f"{prefix} {msg}")
        elif not isinstance(out_id, (RoundPrivateOutId, TablePrivateOutId)):
             unhandled_msg = f"Unhandled PrivateOut Type: Player={player_id}, ID={out_id}, Data={kwargs}"
             print(colorize(unhandled_msg, Colors.BRIGHT_BLACK))
             ai_logger.warning(unhandled_msg)


# ==============================================================================
# Player Action Input Functions
# ==============================================================================

def get_player_action(player_name, to_call, player_money, can_check, can_raise):
    """Prompts the human player for their action and validates the input."""
    prompt_header = colorize(f"--- Your Turn ({player_name}) ---", Colors.BRIGHT_YELLOW + Colors.BOLD)
    print(prompt_header)

    actions = ["FOLD"]
    action_color_map = {
        "FOLD": Colors.BRIGHT_BLACK, "CHECK": Colors.BRIGHT_GREEN,
        "CALL": Colors.BRIGHT_CYAN, "RAISE": Colors.BRIGHT_MAGENTA
    }
    action_parts = [colorize("FOLD", action_color_map["FOLD"])]

    if can_check:
        actions.append("CHECK")
        action_parts.append(colorize("CHECK", action_color_map["CHECK"]))
    elif to_call > 0 and player_money > 0:
        actions.append("CALL")
        effective_call = min(to_call, player_money)
        action_parts.append(colorize(f"CALL({effective_call})", action_color_map["CALL"]))

    if can_raise:
        actions.append("RAISE")
        action_parts.append(colorize("RAISE", action_color_map["RAISE"]))

    print(f"Stack: {colorize(f'${player_money}', Colors.BRIGHT_GREEN)}")
    print(f"Amount to call: {colorize(f'${to_call}', Colors.YELLOW)}")
    print(f"Available actions: {' / '.join(action_parts)}")

    while True:
        try:
            action_str = input(colorize("Enter action: ", Colors.WHITE)).upper().strip()
        except EOFError:
            print(colorize("\nInput ended. Folding.", Colors.RED))
            return RoundPublicInId.FOLD, {}

        if action_str not in actions:
            print(colorize("Invalid action.", Colors.RED) + f" Choose from: {', '.join(actions)}")
            continue

        if action_str == "CALL" and can_check:
            print(colorize("No bet to call. Use CHECK or RAISE.", Colors.YELLOW))
            continue
        if action_str == "CHECK" and not can_check:
            print(colorize(f"Cannot check. Bet is ${to_call}. Use CALL({min(to_call, player_money)}), RAISE, or FOLD.", Colors.YELLOW))
            continue

        if action_str == "FOLD": return RoundPublicInId.FOLD, {}
        if action_str == "CHECK": return RoundPublicInId.CHECK, {}
        if action_str == "CALL": return RoundPublicInId.CALL, {}

        if action_str == "RAISE":
            if not can_raise:
                print(colorize("Error: Raise action is not available.", Colors.RED))
                continue

            min_raise_by = table.big_blind
            max_raise_by = player_money - to_call

            if max_raise_by < min_raise_by and player_money > to_call:
                min_raise_by = max_raise_by

            if max_raise_by <= 0:
                print(colorize("Cannot raise, not enough funds remaining after call.", Colors.RED))
                continue

            while True:
                try:
                    if min_raise_by < max_raise_by: prompt_range = f"(min {min_raise_by}, max {max_raise_by})"
                    else: prompt_range = f"(exactly {max_raise_by} to go all-in)"

                    raise_by_str = input(colorize(f"  Raise BY how much {prompt_range}? ", Colors.WHITE))
                    if not raise_by_str.isdigit(): raise ValueError("Input must be a number.")

                    raise_by = int(raise_by_str)
                    is_all_in_raise = (to_call + raise_by) >= player_money

                    if raise_by <= 0: print(colorize("Raise amount must be positive.", Colors.YELLOW))
                    elif raise_by > max_raise_by: print(colorize(f"Cannot raise by more than your remaining stack allows ({max_raise_by}).", Colors.YELLOW))
                    elif raise_by < min_raise_by and not is_all_in_raise: print(colorize(f"Minimum raise BY amount is {min_raise_by} (unless going all-in).", Colors.YELLOW))
                    else:
                        actual_raise_by = min(raise_by, max_raise_by)
                        return RoundPublicInId.RAISE, {"raise_by": actual_raise_by}

                except ValueError as e: print(colorize(f"Invalid amount: {e}. Please enter a number.", Colors.YELLOW))
                except EOFError:
                    print(colorize("\nInput ended during raise prompt. Folding.", Colors.RED))
                    return RoundPublicInId.FOLD, {}


def get_ai_action(table_state: CommandLineTable, player_id):
    """Gets an action from an AI player using the Together AI API."""
    player_name = table_state._get_player_name(player_id)
    model_name = table_state.ai_model_assignments.get(player_id)

    if not model_name:
        error_msg = f"E: No AI model assigned to player {player_name} (ID: {player_id}). Defaulting to FOLD."
        print(colorize(error_msg, Colors.RED))
        ai_logger.error(error_msg)
        return RoundPublicInId.FOLD, {}

    if VERBOSE:
        model_short_name = AI_MODEL_SHORT_NAMES.get(model_name, model_name.split('/')[-1])
        print(colorize(f"--- AI Thinking ({player_name} using {model_short_name}) ---", Colors.MAGENTA))
    time.sleep(0.5) # Simulate thinking

    prompt = format_state_for_ai(table_state, player_id)
    if "Error:" in prompt:
        error_msg = f"E: Error formatting game state for AI {player_name}: {prompt}. Defaulting to FOLD."
        print(colorize(error_msg, Colors.RED))
        ai_logger.error(error_msg)
        return RoundPublicInId.FOLD, {}

    round_id = table_state.round.id if table_state.round else 0
    history = table_state.ai_message_history[round_id][player_id]
    system_prompt = { "role": "system", "content": "You are a Texas Hold'em poker AI... Respond ONLY with the action name... If raising, add ' AMOUNT: X'..." } # Shortened for brevity
    messages = [system_prompt] + history + [{"role": "user", "content": prompt}]

    ai_logger.info(f"--- AI Turn: {player_name} (Model: {model_name}, Round: {round_id}) ---")
    ai_logger.info(f"Prompt Sent:\n{'-'*20}\n{prompt}\n{'-'*20}")

    ai_response_text = query_together_ai(model_name, messages, AI_TEMPERATURE)

    ai_logger.info(f"Raw Response:\n{'-'*20}\n{ai_response_text or '<< No Response Received >>'}\n{'-'*20}")

    if ai_response_text:
        if VERBOSE: print(colorize(f"AI Raw Response ({player_name}): ", Colors.BRIGHT_BLACK) + f"{ai_response_text}")

        action_enum, action_kwargs = parse_ai_action(ai_response_text)
        player = table_state.seats.getPlayerById(player_id)
        if not player:
             ai_logger.error(f"AI Action: Could not find player object for ID {player_id}. Defaulting to FOLD.")
             return RoundPublicInId.FOLD, {}

        is_possible = False
        fallback_needed = False
        fallback_action = RoundPublicInId.FOLD
        fallback_kwargs = {}
        warn_msg = ""

        if action_enum == RoundPublicInId.FOLD: is_possible = True
        elif action_enum == RoundPublicInId.CHECK:
            is_possible = table_state.can_check
            if not is_possible: warn_msg = f"[AI WARN] AI {player_name} chose CHECK when not possible (To Call: ${table_state.to_call})."
        elif action_enum == RoundPublicInId.CALL:
            is_possible = not table_state.can_check and table_state.to_call > 0 and player.money > 0
            if not is_possible: warn_msg = f"[AI WARN] AI {player_name} chose CALL when not possible or not necessary."
        elif action_enum == RoundPublicInId.RAISE:
            is_possible = table_state.can_raise
            if is_possible:
                raise_by = action_kwargs.get('raise_by', 0)
                max_raise = player.money - table_state.to_call
                min_r = table_state.min_raise
                if max_raise < min_r and player.money > table_state.to_call: min_r = max_raise
                is_all_in_raise = (table_state.to_call + raise_by) >= player.money

                if raise_by <= 0:
                    fallback_needed = True; warn_msg = f"[AI WARN] AI {player_name} chose RAISE with invalid amount {raise_by} (<=0)."
                elif raise_by > max_raise:
                    fallback_needed = True; warn_msg = f"[AI WARN] AI {player_name} chose RAISE {raise_by}, exceeding max possible ({max_raise})."
                    action_kwargs['raise_by'] = max_raise
                    ai_logger.info(f"Corrected AI raise amount to max possible: {max_raise}")
                elif raise_by < min_r and not is_all_in_raise:
                    fallback_needed = True; warn_msg = f"[AI WARN] AI {player_name} chose RAISE {raise_by}, below minimum ({min_r}) and not all-in."
                else: action_kwargs['raise_by'] = min(raise_by, max_raise)
            else:
                 fallback_needed = True; is_possible = False; warn_msg = f"[AI WARN] AI {player_name} chose RAISE when not allowed."

        if not is_possible or fallback_needed:
            if not fallback_needed: pass # warn_msg already set
            print(colorize(warn_msg, Colors.YELLOW)); ai_logger.warning(warn_msg + " Fallback required.")

            if not table_state.can_check and table_state.to_call > 0 and player.money > 0:
                fallback_msg = "Fallback Action: CALL"; fallback_action = RoundPublicInId.CALL; fallback_kwargs = {}
            elif table_state.can_check:
                fallback_msg = "Fallback Action: CHECK"; fallback_action = RoundPublicInId.CHECK; fallback_kwargs = {}
            else:
                fallback_msg = "Fallback Action: FOLD"; fallback_action = RoundPublicInId.FOLD; fallback_kwargs = {}
            print(colorize(fallback_msg, Colors.YELLOW)); ai_logger.info(fallback_msg)
            action_enum = fallback_action; action_kwargs = fallback_kwargs

        history.append({"role": "user", "content": prompt})
        assistant_response_content = f"{action_enum.name}"
        if action_enum == RoundPublicInId.RAISE: assistant_response_content += f" AMOUNT: {action_kwargs['raise_by']}"
        history.append({"role": "assistant", "content": assistant_response_content})

        parsed_action_log = f"Validated Action: {action_enum.name} {action_kwargs}"
        if VERBOSE: print(colorize(f"AI Validated Action ({player_name}): {action_enum.name} {action_kwargs}", Colors.MAGENTA))
        ai_logger.info(parsed_action_log)
        return action_enum, action_kwargs
    else:
        fail_msg = f"AI ({player_name}) failed to provide a response. Defaulting to FOLD."
        print(colorize(fail_msg, Colors.RED)); ai_logger.error(fail_msg)
        return RoundPublicInId.FOLD, {}


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    # --- Initialize Table ---
    table = CommandLineTable(
        _id=0,
        seats=PlayerSeats([None] * NUM_PLAYERS),
        buyin=BUYIN,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND
    )
    ai_logger.info(f"Table initialized with {NUM_PLAYERS} seats. Buy-in: {BUYIN}, Blinds: {SMALL_BLIND}/{BIG_BLIND}.")
    if SHOW_PROBABILITIES:
        ai_logger.info(f"Probability display enabled ({PROBABILITY_SIMULATIONS} simulations).")

    # --- Create Player Objects ---
    players = []
    player_id_counter = 1
    player_ids_generated = set()

    if not AI_ONLY_MODE:
        human_player = Player(table_id=table.id, _id=HUMAN_PLAYER_ID, name=HUMAN_PLAYER_NAME, money=BUYIN)
        players.append(human_player)
        player_ids_generated.add(HUMAN_PLAYER_ID)
        if HUMAN_PLAYER_ID >= player_id_counter: player_id_counter = HUMAN_PLAYER_ID + 1
        ai_logger.info(f"Created Human Player: {HUMAN_PLAYER_NAME} (ID: {HUMAN_PLAYER_ID}) with ${BUYIN}")

    num_ai_to_create = NUM_PLAYERS - len(players)
    ai_logger.info(f"Attempting to create {num_ai_to_create} AI players.")
    if num_ai_to_create > 0 and num_ai_to_create > len(AI_MODEL_LIST):
         warning_msg = f"Warning: Not enough unique models... Models will be reused." # Shortened
         print(colorize(warning_msg, Colors.YELLOW)); ai_logger.warning(warning_msg)
    elif num_ai_to_create > 0 and len(AI_MODEL_LIST) == 0:
         error_msg = f"FATAL: Trying to create {num_ai_to_create} AI players but AI_MODEL_LIST is empty."
         print(colorize(error_msg, Colors.RED)); ai_logger.error(error_msg); sys.exit(1)

    for i in range(num_ai_to_create):
        while player_id_counter in player_ids_generated: player_id_counter += 1
        ai_player_id = player_id_counter; player_ids_generated.add(ai_player_id)
        model_index = i % len(AI_MODEL_LIST)
        model_full_name = AI_MODEL_LIST[model_index]
        ai_short_name = AI_MODEL_SHORT_NAMES.get(model_full_name, f"AI_{ai_player_id}")
        ai_player = Player(table_id=table.id, _id=ai_player_id, name=ai_short_name, money=BUYIN)
        players.append(ai_player)
        ai_logger.info(f"Created AI Player: {ai_short_name} (ID: {ai_player_id})... Will use model: {model_full_name}") # Shortened

    if len(players) != NUM_PLAYERS:
        error_msg = f"FATAL Error: Player object creation mismatch..." # Shortened
        print(colorize(error_msg, Colors.RED)); ai_logger.error(error_msg); sys.exit(1)
    else: ai_logger.info(f"Successfully created {len(players)} player objects.")

    # --- Seat Players ---
    random.shuffle(players)
    ai_logger.info(f"Player seating order after shuffle: {[p.name for p in players]}")
    for p in players:
        table.publicIn(p.id, TablePublicInId.BUYIN, player=p)
        ai_logger.info(f"Sent BUYIN signal for player {p.name} (ID: {p.id})")
    time.sleep(0.1)

    joined_players = table.seats.getPlayerGroup()
    if len(joined_players) != NUM_PLAYERS:
        warn_msg = f"Warning: Player join mismatch... Expected {NUM_PLAYERS}, found {len(joined_players)}." # Shortened
        print(colorize(warn_msg, Colors.YELLOW)); ai_logger.warning(warn_msg)
        if len(joined_players) < 2:
            print(colorize("Error: Less than 2 players joined... Exiting.", Colors.RED)) # Shortened
            ai_logger.error("Exiting due to insufficient joined players."); sys.exit(1)

    table.assign_ai_models() # Assign after seating is confirmed

    # --- Initial Welcome Message ---
    if CLEAR_SCREEN: clear_terminal()
    print(colorize("\n--- Welcome to NLPoker! ---", Colors.BRIGHT_CYAN + Colors.BOLD))
    num_human_configured = 0 if AI_ONLY_MODE else 1
    num_ai_configured = NUM_PLAYERS - num_human_configured
    print(f"{NUM_PLAYERS} players configured ({num_ai_configured} AI, {num_human_configured} Human).")
    print(f"Buy-in: {colorize(f'${BUYIN}', Colors.BRIGHT_GREEN)} each. Blinds: {colorize(f'${SMALL_BLIND}/${BIG_BLIND}', Colors.YELLOW)}")
    if SHOW_PROBABILITIES:
        print(colorize(f"Win probability display enabled ({PROBABILITY_SIMULATIONS} sims/stage).", Colors.BRIGHT_BLUE))
    if AI_ONLY_MODE: print(colorize("Mode: ALL AI Players", Colors.MAGENTA))
    else: print(colorize(f"Mode: Human ({HUMAN_PLAYER_NAME}) vs AI", Colors.MAGENTA))
    ai_logger.info(f"Game Setup: AI_ONLY_MODE={AI_ONLY_MODE}...") # Shortened
    print(colorize("---------------------------------", Colors.BRIGHT_BLACK))
    input(colorize("\nPress Enter to start the game...", Colors.GREEN))

    # --- Main Game Loop ---
    round_count = 0
    try:
        while True:
            active_players_obj = table.seats.getPlayerGroup()
            if len(active_players_obj) < 2:
                print(colorize(f"\nNot enough players ({len(active_players_obj)}) to start... Game over.", Colors.YELLOW + Colors.BOLD)) # Shortened
                ai_logger.warning(f"Game ending: Only {len(active_players_obj)} players remain active.")
                break

            round_count += 1
            initiator = active_players_obj[0]
            ai_logger.info(f"Attempting to start round {round_count} initiated by {initiator.name} (ID: {initiator.id})")

            # Start Round (triggers _newRound internally)
            table.publicIn( initiator.id, TablePublicInId.STARTROUND, round_id=round_count )

            if not table.round:
                ai_logger.error(f"Round {round_count} failed to initialize. Ending game.")
                print(colorize(f"Error: Round {round_count} could not be initialized. Check logs.", Colors.RED))
                break

            # --- Round Action Loop ---
            while table.round and not table.round_over_flag:
                action_player_id_to_process = table._current_action_player_id
                if action_player_id_to_process:
                    table._current_action_player_id = None # Clear flag

                    player = table.seats.getPlayerById(action_player_id_to_process)
                    if not player:
                        print(colorize(f"W: Action requested for invalid Player ID {action_player_id_to_process}. Skipping.", Colors.YELLOW)) # Shortened
                        ai_logger.warning(f"Action requested for missing player ID {action_player_id_to_process} in round {round_count}.")
                        continue

                    current_player_obj_from_lib = None
                    if table.round and hasattr(table.round, "current_player"):
                       try: current_player_obj_from_lib = table.round.current_player
                       except Exception as e: ai_logger.warning(f"Could not get current_player from library state: {e}")

                    if current_player_obj_from_lib and player.id == current_player_obj_from_lib.id:
                        is_ai_turn = AI_ONLY_MODE or player.id != HUMAN_PLAYER_ID
                        if is_ai_turn:
                            action_enum, action_kwargs = get_ai_action(table, player.id)
                        else:
                            # Human action (display happens via publicOut before prompt)
                            action_enum, action_kwargs = get_player_action(
                                player.name, table.to_call, player.money,
                                table.can_check, table.can_raise
                            )
                        table.publicIn(player.id, action_enum, **action_kwargs)

                    elif current_player_obj_from_lib:
                         req_for = f"{player.name}({action_player_id_to_process})"
                         curr_lib = f"{current_player_obj_from_lib.name}({current_player_obj_from_lib.id})"
                         print(colorize(f"W: Action sync issue. Flag={req_for}, Lib={curr_lib}. Wait.", Colors.YELLOW)) # Shortened
                         ai_logger.warning(f"Action request mismatch R{round_count}. Flagged={req_for}, Lib={curr_lib}. Retrying.")
                         table._current_action_player_id = action_player_id_to_process # Re-set to retry
                         time.sleep(0.1)
                    else:
                         print(colorize(f"W: No lib current player when action req for {player.name}. Skip.", Colors.YELLOW)) # Shortened
                         ai_logger.warning(f"No current player R{round_count} when action req for P{action_player_id_to_process}.")

                time.sleep(0.05) # Prevent busy-wait
            # --- End of Round Action Loop ---

            # --- Post-Round Summary ---
            if table.round_over_flag:
                ai_logger.info(f"--- ROUND {round_count} END ---")
                time.sleep(1.0) # Pause for readability

                print(colorize("\nRound ended. Final stacks:", Colors.BRIGHT_WHITE))
                final_players = table.seats.getPlayerGroup()
                if not final_players:
                     print("  No players remaining at the table."); ai_logger.warning("Post-round stack check: No players remaining.")
                else:
                    log_stacks = [f"Stacks after Round {round_count}:"]
                    for p in final_players:
                        p_id = p.id if hasattr(p, 'id') else 'N/A'; money_val = p.money if hasattr(p, 'money') else 'N/A'
                        is_ai_player = AI_ONLY_MODE or (hasattr(p, 'id') and p.id != HUMAN_PLAYER_ID)
                        display_name = p.name if hasattr(p, 'name') else f"P{p_id}"
                        type_indicator_log = " (AI)" if is_ai_player else " (Human)"
                        type_indicator_term = colorize(" (AI)", Colors.MAGENTA) if is_ai_player else colorize(" (Human)", Colors.GREEN)
                        stack_line = f"  - {display_name}{type_indicator_log} (ID:{p_id}): ${money_val}"
                        print(f"  - {colorize(display_name, Colors.CYAN)}{type_indicator_term}: {colorize(f'${money_val}', Colors.BRIGHT_GREEN)}")
                        log_stacks.append(stack_line)
                    ai_logger.info("\n".join(log_stacks))

                if table.round: table.round = None
                table.pending_showdown_messages.clear(); table.pending_winner_messages.clear()

                active_players_check = table.seats.getPlayerGroup()
                if len(active_players_check) < 2:
                     print(colorize("\nNot enough players to continue.", Colors.YELLOW)); break

                if not ALWAYS_CONTINUE:
                    try: cont = input(colorize("\nPlay another round? (y/n): ", Colors.WHITE)).lower().strip()
                    except EOFError: print(colorize("\nInput ended. Exiting.", Colors.RED)); break
                    if cont != 'y': break
                else:
                    print("\nContinuing automatically..."); time.sleep(1)

            else:
                 print(colorize(f"Warning: Round {round_count} loop ended unexpectedly...", Colors.YELLOW)) # Shortened
                 ai_logger.warning(f"Round {round_count} loop ended but round_over_flag was False. Round object: {table.round}")
                 if table.round: table.round = None; break # Assume unrecoverable


    except KeyboardInterrupt:
        print(colorize("\nCtrl+C detected. Exiting game gracefully.", Colors.YELLOW))
        ai_logger.info("Game interrupted by user (Ctrl+C).")
    except Exception as e:
        print(colorize("\n--- UNEXPECTED GAME ERROR ---", Colors.RED + Colors.BOLD))
        traceback.print_exc()
        print(colorize("-----------------------------", Colors.RED + Colors.BOLD))
        ai_logger.exception("UNEXPECTED ERROR in main game loop.")
    finally:
        game_end_msg = "\n--- Game Ended ---"
        print(colorize(game_end_msg, Colors.BRIGHT_CYAN + Colors.BOLD)); ai_logger.info(game_end_msg)

        print(colorize("Final Table Stacks:", Colors.WHITE))
        final_players = table.seats.getPlayerGroup()
        log_stacks = ["Final Stacks (End Game):"]

        if not final_players: print("  No players remaining."); log_stacks.append("  No players remaining.")
        else:
            final_players.sort(key=lambda p: p.money if hasattr(p, 'money') else 0, reverse=True)
            for p in final_players:
                p_id = p.id if hasattr(p, 'id') else 'N/A'; money_val = p.money if hasattr(p, 'money') else 'N/A'; money_str = f"${money_val}"
                is_ai_player = AI_ONLY_MODE or (hasattr(p, 'id') and p.id != HUMAN_PLAYER_ID)
                display_name = p.name if hasattr(p, 'name') else f"P{p_id}"
                type_indicator_log = " (AI)" if is_ai_player else " (Human)"; type_indicator_term = colorize(" (AI)", Colors.MAGENTA) if is_ai_player else colorize(" (Human)", Colors.GREEN)
                stack_line = f"  - {display_name}{type_indicator_log} (ID:{p_id}): {money_str}"
                print(f"  - {colorize(display_name, Colors.CYAN)}{type_indicator_term}: {colorize(money_str, Colors.BRIGHT_GREEN)}")
                log_stacks.append(stack_line)

        ai_logger.info("\n".join(log_stacks))
        print(Colors.RESET)