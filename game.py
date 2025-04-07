# -*- coding: utf-8 -*-
"""
Plays a game of Texas Hold'em poker, allowing for Human vs AI or AI vs AI modes.
Uses the 'pokerlib' library for game logic and the Together AI API for AI opponents.
"""

import sys
import time
import traceback
import os
import requests
import random
import logging
from collections import deque, defaultdict
from dotenv import load_dotenv

# Necessary imports from pokerlib
from pokerlib import Player, PlayerSeats, Table
from pokerlib.enums import ( Hand, Rank, RoundPrivateOutId, RoundPublicInId,
                            RoundPublicOutId, Suit, TablePrivateOutId,
                            TablePublicInId, TablePublicOutId, Turn )

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
AI_TEMPERATURE = 0.7        # Sampling temperature for AI model responses.
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
RANK_MAP = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5", Rank.SIX: "6",
    Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9", Rank.TEN: "T", Rank.JACK: "J",
    Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A"
}
SUIT_MAP = { Suit.SPADE: "♠", Suit.CLUB: "♣", Suit.DIAMOND: "♦", Suit.HEART: "♥" }
SUIT_COLOR_MAP = {
    Suit.SPADE: Colors.WHITE,
    Suit.CLUB: Colors.WHITE,
    Suit.DIAMOND: Colors.BRIGHT_RED,
    Suit.HEART: Colors.BRIGHT_RED
}

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
        rank_str = RANK_MAP.get(rank)
        suit_str = SUIT_MAP.get(suit)
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
        A string representation (e.g., "AS") or "??" if invalid.
    """
    if not card:
        return "??"
    try:
        if hasattr(card, '__len__') and len(card) == 2:
            rank, suit = card
            return f"{RANK_MAP[rank]}{SUIT_MAP[suit]}"
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
    AI player management, game state display, and specific event handling.
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

        # --- AI Management ---
        # Stores message history per round per AI player: {round_id: {player_id: [messages]}}
        self.ai_message_history = defaultdict(lambda: defaultdict(list))
        # Stores AI model assignment: {player_id: 'model_name'}
        self.ai_model_assignments = {}

        # --- End-of-Round Display Buffers ---
        # Buffers messages related to winners to print them together at the end.
        self.pending_winner_messages = []
        # Buffers messages for cards shown during showdown.
        self.pending_showdown_messages = []
        self.printed_showdown_board = False # Flag to ensure board is printed only once during showdown.
        self.showdown_occurred = False      # Flag indicating if any showdown event happened.

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
        and initializes the Round object.

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

        # Clear AI message history for the new round ID if it exists (optional)
        if round_id in self.ai_message_history:
            self.ai_message_history[round_id].clear()

        # Assign AI models if needed (e.g., if players changed or first round)
        # Count how many current players are AI
        num_ai_needed = len([p for p in current_players if AI_ONLY_MODE or (p and hasattr(p, 'id') and p.id != HUMAN_PLAYER_ID)])
        # If there are AI players but no assignments exist, assign them
        if num_ai_needed > 0 and not self.ai_model_assignments :
            self.assign_ai_models()

        # --- Initialize the pokerlib Round ---
        try:
            # Create the actual Round object using the library's mechanism
            self.round = self.RoundClass(
                round_id,
                current_players,
                self.button,
                self.small_blind,
                self.big_blind
            )
            ai_logger.info(f"Successfully initialized pokerlib Round {round_id}.")
        except Exception as e:
            # Catch potential errors during pokerlib's round initialization
            print(colorize(f"--- ROUND INITIALIZATION ERROR (Round {round_id}) ---", Colors.RED))
            print(traceback.format_exc()) # Print detailed error
            ai_logger.error(f"Failed to initialize pokerlib Round {round_id}: {e}", exc_info=True)
            self.round = None # Ensure round object is None if init fails

    def _display_game_state(self, clear_override=False):
        """Displays the current game state to the terminal.

        Includes round info, board cards, pot size, and player details (name, stack,
        cards, status, current bet). Cards are shown for the human player or
        for all non-folded players if the round is over (showdown).

        Args:
            clear_override: If True, bypasses the CLEAR_SCREEN check and does NOT clear.
                           Used for printing final state after round end messages.
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
            # Show basic player list and stacks from the main table seats
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
            # Ensure board cards are valid tuples before formatting
            board_cards_list = [tuple(c) for c in self.round.board if isinstance(c, (list,tuple)) and len(c)==2]
            board_cards_str = format_cards_terminal(board_cards_list)
            pot_total = sum(self.round.pot_size)
            current_turn_enum = self.round.turn
            players_in_round = self.round.players
            # Get blind/button indices safely
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

        for idx, player in enumerate(players_in_round):
            if not player or not hasattr(player, "id"): continue

            is_acting = (player.id == self._current_action_player_id)
            line_prefix = colorize(" > ", Colors.BRIGHT_YELLOW + Colors.BOLD) if is_acting else "   "

            # Player Name and Money
            player_name_str = (player.name if hasattr(player, 'name') else f'P{player.id}').ljust(max_name_len)
            player_name_colored = colorize(player_name_str, Colors.CYAN + (Colors.BOLD if is_acting else ""))
            money_val = player.money if hasattr(player, 'money') else 0
            money_str = colorize(f"${money_val}", Colors.BRIGHT_GREEN)

            # --- Card Display Logic ---
            cards_str = colorize("( ? ? )", Colors.BRIGHT_BLACK) # Default hidden
            show_cards = False

            if player.id in self._player_cards:
                # Check folded status *after* confirming cards exist for the player
                player_folded = hasattr(player, 'is_folded') and player.is_folded

                # Determine if cards should be shown based on mode and game state
                # 1. Show human player's cards (if applicable and not folded)
                if not AI_ONLY_MODE and player.id == HUMAN_PLAYER_ID and not player_folded:
                    show_cards = True
                # 2. Show ANY player's cards in AI-only mode (if not folded) <<< CHANGE HERE
                elif AI_ONLY_MODE and not player_folded:
                    show_cards = True
                # 3. Show cards at showdown (round over and not folded)
                elif self.round_over_flag and not player_folded:
                    show_cards = True
                # Note: Folded players' cards are generally not shown even if known

            if show_cards and player.id in self._player_cards: # Check _player_cards again for safety
                 cards_str = f"( {format_cards_terminal(self._player_cards[player.id])} )"
            elif player_folded: # Ensure folded players *always* show hidden, even if known
                 cards_str = colorize("(FOLDED)", Colors.BRIGHT_BLACK) # Use a clearer "FOLDED" indicator

            # --- End Card Display Logic ---


            # Player Status (Folded, All-in, Blinds, Dealer)
            status = []
            player_round_index = idx
            # Use the cards_str indicator for folded status now
            # if player_folded: status.append(colorize("FOLDED", Colors.BRIGHT_BLACK)) # Removed, handled by cards_str
            if hasattr(player, 'is_all_in') and player.is_all_in: status.append(colorize("ALL-IN", Colors.BRIGHT_RED + Colors.BOLD))
            if player_round_index == button_idx: status.append(colorize("D", Colors.WHITE + Colors.BOLD))
            if player_round_index == sb_idx: status.append(colorize("SB", Colors.YELLOW))
            if player_round_index == bb_idx: status.append(colorize("BB", Colors.YELLOW))
            status_str = " ".join(status)

            # Player's Bet in Current Street
            turn_stake_val = 0
            if ( hasattr(player, "turn_stake") and isinstance(player.turn_stake, list) and
                 hasattr(current_turn_enum, 'value') and
                 len(player.turn_stake) > current_turn_enum.value ):
                turn_stake_val = player.turn_stake[current_turn_enum.value]
            stake_str = colorize(f"[Bet: ${turn_stake_val}]", Colors.MAGENTA) if turn_stake_val > 0 else ""

            # Print player line
            print(f"{line_prefix}{player_name_colored} {money_str.ljust(8)} {cards_str.ljust(20)} {stake_str.ljust(10)} {status_str}")

        print(separator)

    def publicOut(self, out_id, **kwargs):
        """Handles public events broadcast by the poker engine.

        Prints information about game flow (new round, blinds, actions, winners)
        to the console and logs relevant events. Manages display updates and
        buffers showdown/winner information for clarity.

        Args:
            out_id: The ID of the public event (RoundPublicOutId or TablePublicOutId).
            **kwargs: Additional data associated with the event (e.g., player_id, amount).
        """
        player_id = kwargs.get("player_id")
        player_name_raw = self._get_player_name(player_id) if player_id else "System"
        player_name = colorize(player_name_raw, Colors.CYAN)
        msg = ""            # Message to potentially print
        prefix = ""         # Prefix for the message (e.g., [ACTION])
        processed = False   # Flag if the event was handled

        is_round_event = isinstance(out_id, RoundPublicOutId)
        is_table_event = isinstance(out_id, TablePublicOutId)

        # Define player action events that might warrant a pause
        player_action_events = {
            RoundPublicOutId.PLAYERCHECK,
            RoundPublicOutId.PLAYERCALL,
            RoundPublicOutId.PLAYERFOLD,
            RoundPublicOutId.PLAYERRAISE,
            RoundPublicOutId.PLAYERWENTALLIN # Player commits chips
        }

        # --- Round Events ---
        if is_round_event:
            processed = True # Assume handled unless logic dictates otherwise
            prefix = colorize("[ROUND]", Colors.BLUE) # Default prefix for round events
            player = self.seats.getPlayerById(player_id) if player_id else None

            # --- Basic Round Flow & Actions ---
            if out_id == RoundPublicOutId.NEWROUND:
                prefix = colorize("[ROUND]", Colors.BLUE)
                msg = f"Dealing cards for Round {kwargs.get('round_id', '?')}..."
                ai_logger.info(f"Received NEWROUND event (ID: {kwargs.get('round_id', '?')})")
            elif out_id == RoundPublicOutId.NEWTURN:
                prefix = "" # No specific message, handled by state display
                msg = ""
                if not self.round_over_flag: # Avoid redisplay if round just ended
                    self._display_game_state() # Display state at start of Flop, Turn, River
                ai_logger.info(f"Received NEWTURN event: {kwargs.get('turn', 'Unknown')}")
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
            elif out_id == RoundPublicOutId.PLAYERRAISE:
                prefix = colorize("[ACTION]", Colors.BRIGHT_MAGENTA)
                msg = f"{player_name} raises by ${kwargs['raised_by']} (total bet this street: ${kwargs['paid_amount']})"
            elif out_id == RoundPublicOutId.PLAYERISALLIN:
                # This often accompanies another action (call/raise), less critical for pause
                prefix = colorize("[INFO]", Colors.BRIGHT_RED)
                msg = f"{player_name} is {colorize('ALL-IN!', Colors.BOLD)}"
            elif out_id == RoundPublicOutId.PLAYERWENTALLIN:
                # This specifically indicates the action *was* going all-in
                prefix = colorize("[ACTION]", Colors.BRIGHT_RED + Colors.BOLD)
                msg = f"{player_name} goes ALL-IN with ${kwargs['paid_amount']}!"

            # --- Action Required ---
            elif out_id == RoundPublicOutId.PLAYERACTIONREQUIRED:
                 prefix = "" # No message printed here, triggers input prompt/AI
                 msg = ""
                 self._current_action_player_id = player_id
                 # Update state needed for action validation and AI prompt formatting
                 self.to_call = kwargs.get('to_call', 0)
                 self.can_check = self.to_call == 0
                 # Player can raise if they have more money than the call amount
                 self.can_raise = player and hasattr(player, 'money') and player.money > self.to_call
                 self.min_raise = self.big_blind # Reset min raise (can be adjusted later if needed for all-in)
                 ai_logger.info(f"Action required for P{player_id} ({player_name_raw}). To Call: {self.to_call}, Can Check: {self.can_check}, Can Raise: {self.can_raise}")
            elif out_id == RoundPublicOutId.PLAYERCHOICEREQUIRED:
                 # Placeholder for potential future use (e.g., complex side pots)
                 pass

            # --- Showdown Card Display ---
            elif out_id == RoundPublicOutId.PUBLICCARDSHOW:
                prefix = colorize("[SHOWDOWN]", Colors.WHITE)
                msg = "" # Message will be buffered, not printed directly

                # 1. Print Board on the *first* showdown event this round
                if not self.printed_showdown_board and self.round and hasattr(self.round, 'board'):
                    board_cards_list = [tuple(c) for c in self.round.board if isinstance(c, (list,tuple)) and len(c)==2]
                    print() # Blank line for separation
                    print(f"{colorize('[BOARD]', Colors.WHITE)} {format_cards_terminal(board_cards_list)}")
                    self.printed_showdown_board = True # Mark as printed
                    ai_logger.info(f"Showdown: Board printed: {format_cards_for_ai(board_cards_list)}")

                # 2. Format and Buffer the Showdown message
                shown_cards_raw = kwargs.get('cards', [])
                # Ensure cards are converted to tuples for consistency
                shown_cards_tuples = [tuple(c) for c in shown_cards_raw if isinstance(c, (list, tuple)) and len(c)==2]

                if player and shown_cards_tuples:
                    # Store the shown cards (overwrites dealt cards if necessary)
                    self._player_cards[player.id] = tuple(shown_cards_tuples)
                    # Create the message
                    showdown_msg = f"{prefix} {player_name} shows {format_cards_terminal(shown_cards_tuples)}"
                    # Add to buffer only if not already present (avoid duplicates if event fires twice)
                    if showdown_msg not in self.pending_showdown_messages:
                        self.pending_showdown_messages.append(showdown_msg)
                        ai_logger.info(f"Buffered Showdown: {player_name_raw} shows {format_cards_for_ai(shown_cards_tuples)}")
                    self.showdown_occurred = True # Mark that a showdown happened
                else:
                    ai_logger.warning(f"Received PUBLICCARDSHOW for {player_name_raw} but invalid player or cards: {shown_cards_raw}")

            # --- Winner Declaration (Buffering) ---
            elif out_id == RoundPublicOutId.DECLAREPREMATUREWINNER:
                prefix = colorize("[WINNER]", Colors.BRIGHT_YELLOW + Colors.BOLD)
                winner_msg = f"{prefix} {player_name} wins ${kwargs['money_won']} (Prematurely - all others folded)"
                if winner_msg not in self.pending_winner_messages: # Avoid duplicates
                    self.pending_winner_messages.append(winner_msg)
                    ai_logger.info(f"Buffered Premature Winner: {player_name_raw} wins ${kwargs['money_won']}")
                msg = "" # Don't print immediately
            elif out_id == RoundPublicOutId.DECLAREFINISHEDWINNER:
                prefix = colorize("[WINNER]", Colors.BRIGHT_YELLOW + Colors.BOLD)
                hand_name = format_hand_enum(kwargs.get('handname'))
                hand_cards_raw = kwargs.get('handcards', []) # Cards making the hand
                # hand_cards_tuples = [tuple(c) for c in hand_cards_raw if isinstance(c, (list, tuple)) and len(c)==2]
                # cards_display = f" ({format_cards_terminal(hand_cards_tuples)})" if hand_cards_tuples else "" # Optional: show winning hand cards
                winner_msg = f"{prefix} {player_name} wins ${kwargs['money_won']} with {hand_name}" # {cards_display}"
                if winner_msg not in self.pending_winner_messages: # Avoid duplicates
                    self.pending_winner_messages.append(winner_msg)
                    ai_logger.info(f"Buffered Finished Winner: {player_name_raw} wins ${kwargs['money_won']} with {hand_name}")
                msg = "" # Don't print immediately

            # --- Round Finished ---
            elif out_id == RoundPublicOutId.ROUNDFINISHED:
                prefix = "" # No direct message for this event
                msg = ""
                self.round_over_flag = True # Set the flag indicating the round is complete
                ai_logger.info(f"Received ROUNDFINISHED event.")

                # --- Force Showdown Display for Unrevealed Hands ---
                # If a showdown occurred OR only one player remained un-folded,
                # ensure all non-folded players' hands are shown if stored.
                player_ids_with_buffered_showdown = set()
                # Extract player IDs from buffered messages (requires careful parsing of colored strings)
                for buffered_msg in self.pending_showdown_messages:
                    try:
                        # Attempt to find player ID based on the formatted name string
                        name_part = buffered_msg.split(" shows ")[0].split("] ")[-1].strip()
                        found_id = None
                        for p_id, p_obj in self.seats._players.items():
                            if p_obj and colorize(p_obj.name, Colors.CYAN) == name_part:
                                found_id = p_id
                                break
                        if found_id: player_ids_with_buffered_showdown.add(found_id)
                    except Exception as e:
                        ai_logger.warning(f"Could not parse player ID from buffered showdown msg '{buffered_msg}': {e}")

                # --- Print Buffered Showdown & Winner Info ---
                # Print the board if it wasn't printed during a PUBLICCARDSHOW event but a showdown did occur
                if self.showdown_occurred and not self.printed_showdown_board and self.round and hasattr(self.round, 'board'):
                     board_cards_list = [tuple(c) for c in self.round.board if isinstance(c, (list,tuple)) and len(c)==2]
                     print() # Blank line for separation
                     print(f"{colorize('[BOARD]', Colors.WHITE)} {format_cards_terminal(board_cards_list)}")
                     self.printed_showdown_board = True
                     ai_logger.info(f"Showdown: Board printed at ROUNDFINISHED: {format_cards_for_ai(board_cards_list)}")

                # Print all buffered messages now
                for showdown_msg in self.pending_showdown_messages: print(showdown_msg)
                # self.pending_showdown_messages.clear() # Clear after printing
                for winner_msg in self.pending_winner_messages: print(winner_msg)
                # self.pending_winner_messages.clear() # Clear after printing


            # elif out_id == RoundPublicOutId.ROUNDCLOSED: # Less common event
            #     prefix = colorize("[ROUND]", Colors.BLUE)
            #     msg = "Round closed."
            #     self.round_over_flag = True # Also mark round as over


        # --- Table Events ---
        elif is_table_event:
            processed = True
            prefix = colorize("[TABLE]", Colors.MAGENTA)
            if out_id == TablePublicOutId.PLAYERJOINED:
                msg = f"{player_name} joined seat {kwargs['player_seat']}"
                # Re-assign AI models if a player joins mid-game (optional)
                # self.assign_ai_models()
            elif out_id == TablePublicOutId.PLAYERREMOVED:
                 msg = f"{player_name} left the table."
                 # Clean up player data
                 if player_id in self._player_cards: del self._player_cards[player_id]
                 if player_id in self.ai_model_assignments: del self.ai_model_assignments[player_id]
                 if player_id in self.ai_message_history: # Check all rounds
                     for round_hist in self.ai_message_history.values():
                         if player_id in round_hist: del round_hist[player_id]
                 # Consider re-assigning models if needed
                 # self.assign_ai_models()
            elif out_id == TablePublicOutId.NEWROUNDSTARTED:
                prefix = ""; msg = "" # Handled by NEWROUND event from Round class
            elif out_id == TablePublicOutId.ROUNDNOTINITIALIZED:
                prefix = colorize("[ERROR]", Colors.RED); msg = "No round is currently running."
            elif out_id == TablePublicOutId.ROUNDINPROGRESS:
                prefix = colorize("[ERROR]", Colors.RED); msg = "A round is already in progress."
            elif out_id == TablePublicOutId.INCORRECTNUMBEROFPLAYERS:
                prefix = colorize("[ERROR]", Colors.RED); msg = "Incorrect number of players to start (need 2+)."

        # --- Final Print Decision & Logging ---
        should_print = bool(msg) # Print only if msg has content

        # Suppress printing for events handled elsewhere or that provide no useful console output
        if is_round_event and out_id in [
            RoundPublicOutId.NEWTURN,             # Handled by _display_game_state
            RoundPublicOutId.ROUNDFINISHED,       # Handled by printing buffers
            RoundPublicOutId.PUBLICCARDSHOW,      # Buffered
            RoundPublicOutId.DECLAREFINISHEDWINNER, # Buffered
            RoundPublicOutId.DECLAREPREMATUREWINNER,# Buffered
            RoundPublicOutId.PLAYERACTIONREQUIRED, # Handled by input/AI logic trigger
            RoundPublicOutId.PLAYERCHOICEREQUIRED # No current handler
            ]:
             should_print = False
        if is_table_event and out_id == TablePublicOutId.NEWROUNDSTARTED:
             should_print = False # Handled by RoundPublicOutId.NEWROUND

        # Log the raw event regardless of printing
        log_level = logging.WARNING if "ERROR" in prefix else logging.INFO
        # Avoid logging spammy events like NEWTURN or ACTIONREQUIRED unless debugging
        if out_id not in [RoundPublicOutId.NEWTURN, RoundPublicOutId.PLAYERACTIONREQUIRED]:
            ai_logger.log(log_level, f"PublicOut Event: ID={out_id}, Args={kwargs}, Message='{msg}' (Printed: {should_print})")

        # Print the message if required
        if should_print:
            print(f"{prefix} {msg}")
            # --- Add Delay/Pause After AI Actions ---
            # Check if the event was a player action AND the player is an AI
            if is_round_event and out_id in player_action_events:
                is_ai_player = AI_ONLY_MODE or (player_id is not None and player_id != HUMAN_PLAYER_ID)
                if is_ai_player:
                    time.sleep(1.5) # Pause briefly after AI action message for readability

        # --- Log Unhandled Events ---
        elif not processed and out_id != RoundPublicOutId.PLAYERCHOICEREQUIRED:
             unhandled_msg = f"Unhandled PublicOut Event: ID={out_id} Data: {kwargs}"
             print(colorize(unhandled_msg, Colors.BRIGHT_BLACK))
             ai_logger.warning(unhandled_msg)

    def privateOut(self, player_id, out_id, **kwargs):
        """Handles private events sent to a specific player.

        Primarily used for dealing cards and notifying the human player.
        Also handles table/player specific errors like buy-in issues.

        Args:
            player_id: The ID of the player receiving the private event.
            out_id: The ID of the private event (RoundPrivateOutId or TablePrivateOutId).
            **kwargs: Additional data associated with the event (e.g., cards).
        """
        player_name_raw = self._get_player_name(player_id)
        player_name_color = colorize(player_name_raw, Colors.CYAN) # Colored name for potential printing
        prefix = colorize(f"[PRIVATE to {player_name_raw}]", Colors.YELLOW)
        msg = ""
        log_msg = f"PrivateOut for P{player_id} ({player_name_raw}): ID={out_id}, Args={kwargs}"
        log_level = logging.INFO

        if isinstance(out_id, RoundPrivateOutId):
            if out_id == RoundPrivateOutId.DEALTCARDS:
                cards_raw = kwargs.get('cards')
                if cards_raw and isinstance(cards_raw, (list, tuple)) and len(cards_raw) == 2:
                    # Ensure cards are stored as tuple of tuples: ((R, S), (R, S))
                    cards_tuples = tuple(tuple(c) for c in cards_raw if isinstance(c, (list, tuple)) and len(c)==2)
                    if len(cards_tuples) == 2:
                        self._player_cards[player_id] = cards_tuples # Store the dealt cards
                        card_str_terminal = format_cards_terminal(cards_tuples)
                        card_str_log = format_cards_for_ai(cards_tuples)
                        msg = f"You are dealt {card_str_terminal}"
                        log_msg += f" - Cards: {card_str_log}"
                    else:
                        msg = colorize("Card data conversion error.", Colors.RED)
                        log_msg += " - Error: Could not convert raw cards to tuples."
                        log_level = logging.ERROR
                else:
                    msg = colorize("Card dealing error (invalid data received).", Colors.RED)
                    log_msg += f" - Error: Invalid card data received: {cards_raw}"
                    log_level = logging.ERROR

        elif isinstance(out_id, TablePrivateOutId):
            prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED) # Most table private outs are errors
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
            else: # Handle potential future/unknown table private events
                 prefix = colorize(f"[UNHANDLED PRIVATE to {player_name_raw}]", Colors.BRIGHT_BLACK)
                 msg = f"Unknown TablePrivateOutId: {out_id} Data: {kwargs}"
                 log_level = logging.WARNING


        # --- Logging ---
        ai_logger.log(log_level, log_msg)

        # --- Print to Console (Only for Human Player in Mixed Mode) ---
        if msg and not AI_ONLY_MODE and player_id == HUMAN_PLAYER_ID:
            print(f"{prefix} {msg}")
        # --- Handle Unhandled Enums ---
        elif not isinstance(out_id, (RoundPrivateOutId, TablePrivateOutId)):
             unhandled_msg = f"Unhandled PrivateOut Type: Player={player_id}, ID={out_id}, Data={kwargs}"
             print(colorize(unhandled_msg, Colors.BRIGHT_BLACK))
             ai_logger.warning(unhandled_msg)


# ==============================================================================
# Player Action Input Functions
# ==============================================================================

def get_player_action(player_name, to_call, player_money, can_check, can_raise):
    """Prompts the human player for their action and validates the input.

    Args:
        player_name: The name of the human player.
        to_call: The amount the player needs to call to stay in the hand.
        player_money: The player's current stack size.
        can_check: Boolean indicating if checking is a valid action.
        can_raise: Boolean indicating if raising is a valid action.

    Returns:
        A tuple containing:
        - action_enum: The corresponding RoundPublicInId enum member.
        - action_kwargs: A dictionary with action parameters (e.g., {'raise_by': amount}).
    """
    prompt_header = colorize(f"--- Your Turn ({player_name}) ---", Colors.BRIGHT_YELLOW + Colors.BOLD)
    print(prompt_header)

    # Build available actions list and display string
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
        effective_call = min(to_call, player_money) # Show actual amount needed (or all-in)
        action_parts.append(colorize(f"CALL({effective_call})", action_color_map["CALL"]))

    if can_raise:
        actions.append("RAISE")
        action_parts.append(colorize("RAISE", action_color_map["RAISE"]))

    print(f"Stack: {colorize(f'${player_money}', Colors.BRIGHT_GREEN)}")
    print(f"Amount to call: {colorize(f'${to_call}', Colors.YELLOW)}")
    print(f"Available actions: {' / '.join(action_parts)}")

    # Input loop
    while True:
        try:
            action_str = input(colorize("Enter action: ", Colors.WHITE)).upper().strip()
        except EOFError:
            # Handle Ctrl+D or end of input stream
            print(colorize("\nInput ended. Folding.", Colors.RED))
            return RoundPublicInId.FOLD, {}

        # --- Action Validation ---
        if action_str not in actions:
            print(colorize("Invalid action.", Colors.RED) + f" Choose from: {', '.join(actions)}")
            continue

        # Prevent CHECK when CALL is required or vice-versa
        if action_str == "CALL" and can_check:
            print(colorize("No bet to call. Use CHECK or RAISE.", Colors.YELLOW))
            continue
        if action_str == "CHECK" and not can_check:
            print(colorize(f"Cannot check. Bet is ${to_call}. Use CALL({min(to_call, player_money)}), RAISE, or FOLD.", Colors.YELLOW))
            continue

        # --- Handle Specific Actions ---
        if action_str == "FOLD":
            return RoundPublicInId.FOLD, {}

        if action_str == "CHECK":
            if can_check:
                return RoundPublicInId.CHECK, {}
            else: # Should be caught above, but double-check
                print(colorize(f"Cannot check. Bet is ${to_call}.", Colors.YELLOW))
                continue

        if action_str == "CALL":
            if not can_check and to_call > 0 and player_money > 0:
                return RoundPublicInId.CALL, {}
            else: # Should be caught above, but double-check
                 print(colorize("Cannot call. Use CHECK or FOLD.", Colors.YELLOW))
                 continue

        if action_str == "RAISE":
            if not can_raise: # Should be caught by actions list, but double-check
                print(colorize("Error: Raise action is not available.", Colors.RED))
                continue

            # Determine raise constraints
            min_raise_by = table.big_blind # Base minimum raise amount
            max_raise_by = player_money - to_call # Max additional amount player can bet

            # Special case: If the only possible raise is all-in and less than a standard min raise
            if max_raise_by < min_raise_by and player_money > to_call:
                min_raise_by = max_raise_by # The minimum raise *possible* is going all-in

            # Cannot raise if max raise amount is zero or less (shouldn't happen if can_raise is true)
            if max_raise_by <= 0:
                print(colorize("Cannot raise, not enough funds remaining after call.", Colors.RED))
                continue

            # Prompt for raise amount loop
            while True:
                try:
                    # Provide clear min/max guidance
                    if min_raise_by < max_raise_by:
                         prompt_range = f"(min {min_raise_by}, max {max_raise_by})"
                    else: # Only possible raise is exactly max_raise_by (all-in)
                         prompt_range = f"(exactly {max_raise_by} to go all-in)"

                    raise_by_str = input(colorize(f"  Raise BY how much {prompt_range}? ", Colors.WHITE))
                    if not raise_by_str.isdigit():
                        raise ValueError("Input must be a number.")

                    raise_by = int(raise_by_str)
                    is_all_in_raise = (to_call + raise_by) >= player_money # Check if this raise results in all-in

                    # Validate raise amount
                    if raise_by <= 0:
                        print(colorize("Raise amount must be positive.", Colors.YELLOW))
                    elif raise_by > max_raise_by:
                        print(colorize(f"Cannot raise by more than your remaining stack allows ({max_raise_by}).", Colors.YELLOW))
                    # Allow raises smaller than standard big blind min_raise ONLY if it results in all-in
                    elif raise_by < min_raise_by and not is_all_in_raise:
                         print(colorize(f"Minimum raise BY amount is {min_raise_by} (unless going all-in).", Colors.YELLOW))
                    else:
                        # Valid raise amount
                        # Ensure we don't somehow exceed max_raise_by due to input weirdness
                        actual_raise_by = min(raise_by, max_raise_by)
                        return RoundPublicInId.RAISE, {"raise_by": actual_raise_by}

                except ValueError as e:
                    print(colorize(f"Invalid amount: {e}. Please enter a number.", Colors.YELLOW))
                except EOFError:
                    print(colorize("\nInput ended during raise prompt. Folding.", Colors.RED))
                    return RoundPublicInId.FOLD, {}


def get_ai_action(table_state: CommandLineTable, player_id):
    """Gets an action from an AI player using the Together AI API.

    Formats the game state, queries the assigned AI model, parses the response,
    validates the action against game rules, and applies fallbacks if necessary.

    Args:
        table_state: The CommandLineTable instance holding the game state.
        player_id: The ID of the AI player whose turn it is.

    Returns:
        A tuple containing:
        - action_enum: The validated RoundPublicInId enum member for the AI's action.
        - action_kwargs: A dictionary with action parameters.
        Defaults to FOLD if the API call fails or validation requires it.
    """
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
    time.sleep(0.5) # Small delay to simulate thinking

    # --- Format Prompt ---
    prompt = format_state_for_ai(table_state, player_id)
    if "Error:" in prompt:
        # Handle errors during state formatting
        error_msg = f"E: Error formatting game state for AI {player_name}: {prompt}. Defaulting to FOLD."
        print(colorize(error_msg, Colors.RED))
        ai_logger.error(error_msg)
        return RoundPublicInId.FOLD, {}

    # --- Prepare API Call ---
    round_id = table_state.round.id if table_state.round else 0
    # Retrieve message history for this AI in this round
    history = table_state.ai_message_history[round_id][player_id]

    # Define the system prompt instructing the AI
    system_prompt = {
        "role": "system",
        "content": "You are a Texas Hold'em poker AI. Analyze the game state and decide your next action. Focus on making a reasonable poker play based on your cards, the board, pot size, and opponent actions. Respond ONLY with the action name (FOLD, CHECK, CALL, RAISE). If raising, add ' AMOUNT: X' on the same line, where X is the integer amount to raise BY (the additional chips beyond the call amount). Do not add any other explanation, commentary, or text."
    }
    messages = [system_prompt] + history + [{"role": "user", "content": prompt}]

    # Log the prompt being sent
    ai_logger.info(f"--- AI Turn: {player_name} (Model: {model_name}, Round: {round_id}) ---")
    ai_logger.info(f"Prompt Sent:\n{'-'*20}\n{prompt}\n{'-'*20}")

    # --- Query AI ---
    ai_response_text = query_together_ai(model_name, messages, AI_TEMPERATURE)

    # Log the raw response
    ai_logger.info(f"Raw Response:\n{'-'*20}\n{ai_response_text or '<< No Response Received >>'}\n{'-'*20}")

    if ai_response_text:
        if VERBOSE:
            print(colorize(f"AI Raw Response ({player_name}): ", Colors.BRIGHT_BLACK) + f"{ai_response_text}")

        # --- Parse and Validate Action ---
        action_enum, action_kwargs = parse_ai_action(ai_response_text)
        player = table_state.seats.getPlayerById(player_id)
        if not player: # Should not happen if player_id is valid, but check anyway
             ai_logger.error(f"AI Action: Could not find player object for ID {player_id}. Defaulting to FOLD.")
             return RoundPublicInId.FOLD, {}

        is_possible = False         # Can the AI *legally* perform the chosen action?
        fallback_needed = False     # Does the action need to be changed (e.g., invalid raise amount)?
        fallback_action = RoundPublicInId.FOLD # Default fallback
        fallback_kwargs = {}
        warn_msg = "" # Store warning message if fallback occurs

        # Check legality based on current game state flags
        if action_enum == RoundPublicInId.FOLD:
            is_possible = True
        elif action_enum == RoundPublicInId.CHECK:
            is_possible = table_state.can_check
            if not is_possible: warn_msg = f"[AI WARN] AI {player_name} chose CHECK when not possible (To Call: ${table_state.to_call})."
        elif action_enum == RoundPublicInId.CALL:
            # Possible if not allowed to check, there's a bet to call, and player has money
            is_possible = not table_state.can_check and table_state.to_call > 0 and player.money > 0
            if not is_possible: warn_msg = f"[AI WARN] AI {player_name} chose CALL when not possible or not necessary."
        elif action_enum == RoundPublicInId.RAISE:
            is_possible = table_state.can_raise
            if is_possible:
                # Further validation for the raise amount
                raise_by = action_kwargs.get('raise_by', 0)
                max_raise = player.money - table_state.to_call
                min_r = table_state.min_raise
                # Adjust min raise if only possible raise is all-in < standard min
                if max_raise < min_r and player.money > table_state.to_call:
                    min_r = max_raise
                is_all_in_raise = (table_state.to_call + raise_by) >= player.money

                if raise_by <= 0:
                    fallback_needed = True
                    warn_msg = f"[AI WARN] AI {player_name} chose RAISE with invalid amount {raise_by} (<=0)."
                elif raise_by > max_raise:
                    fallback_needed = True
                    warn_msg = f"[AI WARN] AI {player_name} chose RAISE {raise_by}, exceeding max possible ({max_raise})."
                    action_kwargs['raise_by'] = max_raise # Correct the raise amount to max possible
                    ai_logger.info(f"Corrected AI raise amount to max possible: {max_raise}")
                # Allow raise < min_r ONLY if it results in all-in
                elif raise_by < min_r and not is_all_in_raise:
                    fallback_needed = True
                    warn_msg = f"[AI WARN] AI {player_name} chose RAISE {raise_by}, below minimum ({min_r}) and not all-in."
                else:
                    # Raise amount seems valid within constraints
                    action_kwargs['raise_by'] = min(raise_by, max_raise) # Ensure clipped to max
            else: # AI chose RAISE when table_state.can_raise was False
                 fallback_needed = True # Need fallback, but action itself was impossible
                 is_possible = False # Mark as impossible
                 warn_msg = f"[AI WARN] AI {player_name} chose RAISE when not allowed."

        # --- Apply Fallback Logic ---
        if not is_possible or fallback_needed:
            if not fallback_needed: # If action was impossible but amount wasn't the issue (e.g., CHECK when must CALL)
                 # warn_msg is already set from the checks above
                 pass

            print(colorize(warn_msg, Colors.YELLOW))
            ai_logger.warning(warn_msg + " Fallback required.")

            # Determine the safest fallback action:
            # 1. CALL: If a call is required and possible.
            # 2. CHECK: If checking is possible.
            # 3. FOLD: Otherwise.
            if not table_state.can_check and table_state.to_call > 0 and player.money > 0:
                fallback_msg = "Fallback Action: CALL"
                fallback_action = RoundPublicInId.CALL
                fallback_kwargs = {}
            elif table_state.can_check:
                fallback_msg = "Fallback Action: CHECK"
                fallback_action = RoundPublicInId.CHECK
                fallback_kwargs = {}
            else:
                fallback_msg = "Fallback Action: FOLD"
                fallback_action = RoundPublicInId.FOLD
                fallback_kwargs = {}

            print(colorize(fallback_msg, Colors.YELLOW))
            ai_logger.info(fallback_msg)
            action_enum = fallback_action
            action_kwargs = fallback_kwargs

        # --- Log History and Final Action ---
        # Store interaction in history *after* validation/fallback
        history.append({"role": "user", "content": prompt})
        # Store the *validated* action the AI is taking
        assistant_response_content = f"{action_enum.name}"
        if action_enum == RoundPublicInId.RAISE:
            assistant_response_content += f" AMOUNT: {action_kwargs['raise_by']}"
        history.append({"role": "assistant", "content": assistant_response_content})

        # Log the final validated action
        parsed_action_log = f"Validated Action: {action_enum.name} {action_kwargs}"
        if VERBOSE:
            print(colorize(f"AI Validated Action ({player_name}): {action_enum.name} {action_kwargs}", Colors.MAGENTA))
        ai_logger.info(parsed_action_log)

        return action_enum, action_kwargs

    else:
        # Handle API call failure or no response
        fail_msg = f"AI ({player_name}) failed to provide a response. Defaulting to FOLD."
        print(colorize(fail_msg, Colors.RED))
        ai_logger.error(fail_msg)
        return RoundPublicInId.FOLD, {}


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    # --- Initialize Table ---
    # PlayerSeats expects a list of Nones initially, size determines max players
    table = CommandLineTable(
        _id=0,
        seats=PlayerSeats([None] * NUM_PLAYERS),
        buyin=BUYIN,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND
    )
    ai_logger.info(f"Table initialized with {NUM_PLAYERS} seats. Buy-in: {BUYIN}, Blinds: {SMALL_BLIND}/{BIG_BLIND}.")

    # --- Create Player Objects ---
    players = []
    player_id_counter = 1 # Start generating IDs from 1
    player_ids_generated = set()

    # 1. Create Human Player (if not AI_ONLY_MODE)
    if not AI_ONLY_MODE:
        human_player = Player(table_id=table.id, _id=HUMAN_PLAYER_ID, name=HUMAN_PLAYER_NAME, money=BUYIN)
        players.append(human_player)
        player_ids_generated.add(HUMAN_PLAYER_ID)
        # Ensure the counter starts *after* the human ID to avoid collision
        if HUMAN_PLAYER_ID >= player_id_counter:
            player_id_counter = HUMAN_PLAYER_ID + 1
        ai_logger.info(f"Created Human Player: {HUMAN_PLAYER_NAME} (ID: {HUMAN_PLAYER_ID}) with ${BUYIN}")

    # 2. Create AI Players
    num_ai_to_create = NUM_PLAYERS - len(players) # Calculate remaining players needed
    ai_logger.info(f"Attempting to create {num_ai_to_create} AI players.")

    # Check if enough models are defined
    if num_ai_to_create > 0 and num_ai_to_create > len(AI_MODEL_LIST):
         warning_msg = f"Warning: Not enough unique models in AI_MODEL_LIST ({len(AI_MODEL_LIST)}) for {num_ai_to_create} AI players. Models will be reused."
         print(colorize(warning_msg, Colors.YELLOW))
         ai_logger.warning(warning_msg)
    elif num_ai_to_create > 0 and len(AI_MODEL_LIST) == 0:
         # This case should ideally be caught by earlier checks, but safeguard here
         error_msg = f"FATAL: Trying to create {num_ai_to_create} AI players but AI_MODEL_LIST is empty."
         print(colorize(error_msg, Colors.RED))
         ai_logger.error(error_msg)
         sys.exit(1)


    for i in range(num_ai_to_create):
        # Find the next available unique ID, skipping the human ID if necessary
        while player_id_counter in player_ids_generated:
            player_id_counter += 1
        ai_player_id = player_id_counter
        player_ids_generated.add(ai_player_id)

        # Assign model and get short name, cycling through the list
        # Use index `i` relative to the number of AIs being created for model cycling
        model_index = i % len(AI_MODEL_LIST)
        model_full_name = AI_MODEL_LIST[model_index]
        # Use short name for the Player object's name for clarity in display
        ai_short_name = AI_MODEL_SHORT_NAMES.get(model_full_name, f"AI_{ai_player_id}") # Fallback name

        ai_player = Player(table_id=table.id, _id=ai_player_id, name=ai_short_name, money=BUYIN)
        players.append(ai_player)
        ai_logger.info(f"Created AI Player: {ai_short_name} (ID: {ai_player_id}) with ${BUYIN}. Will use model: {model_full_name}")
        # player_id_counter increment is handled by the while loop

    # Final check on the number of player objects created vs configured
    if len(players) != NUM_PLAYERS:
        error_msg = f"FATAL Error: Player object creation mismatch. Expected {NUM_PLAYERS}, created {len(players)}. Check IDs and AI_ONLY_MODE."
        print(colorize(error_msg, Colors.RED))
        ai_logger.error(error_msg + f" Generated IDs: {player_ids_generated}")
        sys.exit(1)
    else:
        ai_logger.info(f"Successfully created {len(players)} player objects.")

    # --- Seat Players ---
    random.shuffle(players) # Shuffle seating order *before* joining
    ai_logger.info(f"Player seating order after shuffle: {[p.name for p in players]}")

    # Join players to the table *after* all are created and shuffled
    for p in players:
        # Send the signal for the player to join/buy-in
        # The actual confirmation message '[TABLE] Player joined seat X' will be printed by table.publicOut
        table.publicIn(p.id, TablePublicInId.BUYIN, player=p)
        ai_logger.info(f"Sent BUYIN signal for player {p.name} (ID: {p.id})")

    # Brief pause to allow the library to process join events before proceeding
    time.sleep(0.1)

    # Verify players actually joined the table seats (important!)
    joined_players = table.seats.getPlayerGroup()
    if len(joined_players) != NUM_PLAYERS:
        warn_msg = f"Warning: Player join mismatch after BUYIN. Expected {NUM_PLAYERS}, found {len(joined_players)} actually seated. Game might fail."
        print(colorize(warn_msg, Colors.YELLOW))
        ai_logger.warning(warn_msg + f" Joined Player IDs: {[p.id for p in joined_players]}")
        # Optional: Exit if too few players joined successfully
        if len(joined_players) < 2:
            print(colorize("Error: Less than 2 players joined successfully. Cannot start game. Exiting.", Colors.RED))
            ai_logger.error("Exiting due to insufficient joined players.")
            sys.exit(1)

    # Assign AI Models *after* players have successfully joined and are in seats
    # (assign_ai_models uses table.seats.getPlayerGroup())
    table.assign_ai_models()


    # --- Initial Welcome Message ---
    # Clear terminal once at the very beginning if enabled
    if CLEAR_SCREEN: clear_terminal()
    print(colorize("\n--- Welcome to NLPoker! ---", Colors.BRIGHT_CYAN + Colors.BOLD))
    # Display counts based on configuration for clarity
    num_human_configured = 0 if AI_ONLY_MODE else 1
    num_ai_configured = NUM_PLAYERS - num_human_configured
    print(f"{NUM_PLAYERS} players configured ({num_ai_configured} AI, {num_human_configured} Human).")
    print(f"Buy-in: {colorize(f'${BUYIN}', Colors.BRIGHT_GREEN)} each. Blinds: {colorize(f'${SMALL_BLIND}/${BIG_BLIND}', Colors.YELLOW)}")
    if AI_ONLY_MODE:
        print(colorize("Mode: ALL AI Players", Colors.MAGENTA))
    else:
        print(colorize(f"Mode: Human ({HUMAN_PLAYER_NAME}) vs AI", Colors.MAGENTA))
    ai_logger.info(f"Game Setup: AI_ONLY_MODE={AI_ONLY_MODE}. Human ID={HUMAN_PLAYER_ID if not AI_ONLY_MODE else 'N/A'}. Num AI Configured={num_ai_configured}. Total Players Configured={NUM_PLAYERS}.")
    print(colorize("---------------------------------", Colors.BRIGHT_BLACK))
    input(colorize("\nPress Enter to start the game...", Colors.GREEN)) # Wait for user input

    # --- Main Game Loop ---
    round_count = 0
    try:
        while True:
            # --- Pre-Round Check ---
            # Check active players *before* starting the round
            active_players_obj = table.seats.getPlayerGroup()
            if len(active_players_obj) < 2:
                print(colorize(f"\nNot enough players ({len(active_players_obj)}) to start a new round. Game over.", Colors.YELLOW + Colors.BOLD))
                ai_logger.warning(f"Game ending: Only {len(active_players_obj)} players remain active.")
                break # Exit the main game loop

            round_count += 1
            # Use the first active player as the initiator (library handles button rotation)
            initiator = active_players_obj[0]
            ai_logger.info(f"Attempting to start round {round_count} initiated by {initiator.name} (ID: {initiator.id})")

            # --- Start Round ---
            # Reset round-specific flags within the table object before starting
            # (This is now handled robustly inside _newRound)
            # table.round_over_flag = False
            # table.printed_showdown_board = False
            # table.showdown_occurred = False
            # table.pending_winner_messages.clear()
            # table.pending_showdown_messages.clear()

            # Trigger the library to start a new round
            # _newRound will be called internally by the library via this input
            table.publicIn( initiator.id, TablePublicInId.STARTROUND, round_id=round_count )

            # Check if round initialization failed in _newRound
            if not table.round:
                ai_logger.error(f"Round {round_count} failed to initialize. Ending game.")
                print(colorize(f"Error: Round {round_count} could not be initialized. Check logs.", Colors.RED))
                break

            # --- Round Action Loop ---
            # Continue as long as the round object exists and the round_over_flag is not set
            while table.round and not table.round_over_flag:
                # Check if an action is required (flag set by publicOut)
                action_player_id_to_process = table._current_action_player_id
                if action_player_id_to_process:
                    table._current_action_player_id = None # Clear flag immediately to prevent reprocessing

                    player = table.seats.getPlayerById(action_player_id_to_process)
                    if not player:
                        # Should not happen if ID was valid, but handle defensively
                        print(colorize(f"W: Action requested for invalid or missing Player ID {action_player_id_to_process}. Skipping turn.", Colors.YELLOW))
                        ai_logger.warning(f"Action requested for missing player ID {action_player_id_to_process} in round {round_count}.")
                        continue # Skip this action request

                    # --- Verify Turn Synchronization ---
                    # Double check if it's actually this player's turn according to the library's internal state
                    # This helps catch potential race conditions or state mismatches.
                    current_player_obj_from_lib = None
                    if table.round and hasattr(table.round, "current_player"):
                       try:
                           current_player_obj_from_lib = table.round.current_player
                       except Exception as e:
                           # This might fail if the round ended unexpectedly between checks
                           ai_logger.warning(f"Could not get current_player from library state: {e}")
                           pass

                    if current_player_obj_from_lib and player.id == current_player_obj_from_lib.id:
                        # State matches: Process the action for this player

                        # Optional: Check if player state seems ready (has attributes needed)
                        # This is a deeper check that might be overly cautious
                        # if not all(hasattr(player, a) for a in ['money','stake','turn_stake']):
                        #      print(colorize(f"W: Player state not fully initialized for {player.name}. Retrying briefly.", Colors.YELLOW));
                        #      ai_logger.warning(f"Player state potentially not ready for {player.name} (ID: {player.id}) in round {round_count}. Retrying.")
                        #      time.sleep(0.1);
                        #      table._current_action_player_id = action_player_id_to_process # Re-set flag to retry once
                        #      continue

                        # Determine if Human or AI turn
                        is_ai_turn = AI_ONLY_MODE or player.id != HUMAN_PLAYER_ID
                        if is_ai_turn:
                            # Get action from AI
                            action_enum, action_kwargs = get_ai_action(table, player.id)
                        else:
                            # Get action from Human
                            # Display state *before* prompting human player
                            action_enum, action_kwargs = get_player_action(
                                player.name, table.to_call, player.money,
                                table.can_check, table.can_raise
                            )

                        # Send the chosen action back into the poker engine
                        table.publicIn(player.id, action_enum, **action_kwargs)

                    elif current_player_obj_from_lib:
                         # State mismatch: Our flag indicates one player, library indicates another
                         req_for = f"{player.name}(ID:{action_player_id_to_process})"
                         curr_lib = f"{current_player_obj_from_lib.name}(ID:{current_player_obj_from_lib.id})"
                         print(colorize(f"W: Action request sync issue. Flag for={req_for}, Library current={curr_lib}. Waiting.", Colors.YELLOW))
                         ai_logger.warning(f"Action request mismatch in round {round_count}. Flagged={req_for}, LibraryCurrent={curr_lib}. Re-setting flag and delaying.")
                         # Re-set the flag to ensure the original request is eventually processed or superseded
                         table._current_action_player_id = action_player_id_to_process
                         time.sleep(0.1) # Brief pause before next loop iteration
                    else:
                         # Library doesn't have a current player (round might have ended?)
                         print(colorize(f"W: No current player in library state when action requested for {player.name}. Skipping.", Colors.YELLOW))
                         ai_logger.warning(f"No current player in library state when action requested for P{action_player_id_to_process} in round {round_count}. Round may have ended.")
                         # Do not re-set the flag, assume the action is now moot

                # Small sleep to prevent busy-waiting and allow library processing time
                time.sleep(0.05)
            # --- End of Round Action Loop ---


            # --- Post-Round Summary ---
            # This block executes after the inner `while table.round and not table.round_over_flag:` loop finishes.
            # Check if the round *actually* finished normally (flag set by ROUNDFINISHED event)
            if table.round_over_flag:
                ai_logger.info(f"--- ROUND {round_count} END ---")

                # Board/Showdown/Winner messages should have been printed by publicOut
                # when the ROUNDFINISHED event was handled.

                # Optional delay for readability before showing final stacks
                time.sleep(1.0) # Adjust delay as needed

                # Print final stacks for the round
                print(colorize("\nRound ended. Final stacks:", Colors.BRIGHT_WHITE))
                final_players = table.seats.getPlayerGroup() # Get current players at table
                if not final_players:
                     print("  No players remaining at the table.")
                     ai_logger.warning("Post-round stack check: No players remaining.")
                else:
                    log_stacks = [f"Stacks after Round {round_count}:"]
                    for p in final_players:
                        p_id = p.id if hasattr(p, 'id') else 'N/A'
                        money_val = p.money if hasattr(p, 'money') else 'N/A'
                        is_ai_player = AI_ONLY_MODE or (hasattr(p, 'id') and p.id != HUMAN_PLAYER_ID)
                        display_name = p.name if hasattr(p, 'name') else f"P{p_id}"
                        # Use consistent indicators for log vs terminal
                        type_indicator_log = " (AI)" if is_ai_player else " (Human)"
                        type_indicator_term = colorize(" (AI)", Colors.MAGENTA) if is_ai_player else colorize(" (Human)", Colors.GREEN)
                        stack_line = f"  - {display_name}{type_indicator_log} (ID:{p_id}): ${money_val}"
                        print(f"  - {colorize(display_name, Colors.CYAN)}{type_indicator_term}: {colorize(f'${money_val}', Colors.BRIGHT_GREEN)}")
                        log_stacks.append(stack_line)
                    ai_logger.info("\n".join(log_stacks))

                # Cleanly reset the library's round object reference
                if table.round: table.round = None
                # Clear buffers just in case, though should be cleared in publicOut
                table.pending_showdown_messages.clear()
                table.pending_winner_messages.clear()

                # --- Ask to Continue / Auto-Continue ---
                # Check for sufficient players *before* asking/continuing
                active_players_check = table.seats.getPlayerGroup()
                if len(active_players_check) < 2:
                     print(colorize("\nNot enough players to continue.", Colors.YELLOW))
                     break # Exit outer while loop

                if not ALWAYS_CONTINUE:
                    try:
                        cont = input(colorize("\nPlay another round? (y/n): ", Colors.WHITE)).lower().strip()
                    except EOFError:
                        print(colorize("\nInput ended. Exiting.", Colors.RED))
                        break # Exit outer while loop
                    if cont != 'y':
                        break # Exit outer while loop
                else: # ALWAYS_CONTINUE is True
                    print("\nContinuing automatically...")
                    time.sleep(1)
                    # clear_terminal() # Optionally clear before next round starts

            else:
                 # This case means the inner loop exited but round_over_flag wasn't set
                 # This might indicate an error or unexpected state in the library.
                 print(colorize(f"Warning: Round {round_count} loop ended unexpectedly (round_over_flag not set).", Colors.YELLOW))
                 ai_logger.warning(f"Round {round_count} loop ended but round_over_flag was False. Round object: {table.round}")
                 if table.round: table.round = None # Attempt cleanup
                 # Consider breaking the main loop here if this state is unrecoverable
                 # break


    except KeyboardInterrupt:
        print(colorize("\nCtrl+C detected. Exiting game gracefully.", Colors.YELLOW))
        ai_logger.info("Game interrupted by user (Ctrl+C).")
    except Exception as e:
        # Catch any other unexpected errors in the main loop
        print(colorize("\n--- UNEXPECTED GAME ERROR ---", Colors.RED + Colors.BOLD))
        traceback.print_exc() # Print detailed traceback to console
        print(colorize("-----------------------------", Colors.RED + Colors.BOLD))
        ai_logger.exception("UNEXPECTED ERROR in main game loop.") # Log exception with traceback
    finally:
        # --- Final Game End Summary ---
        game_end_msg = "\n--- Game Ended ---"
        print(colorize(game_end_msg, Colors.BRIGHT_CYAN + Colors.BOLD))
        ai_logger.info(game_end_msg)

        print(colorize("Final Table Stacks:", Colors.WHITE))
        final_players = table.seats.getPlayerGroup()
        log_stacks = ["Final Stacks (End Game):"]

        if not final_players:
            print("  No players remaining at the table.")
            log_stacks.append("  No players remaining.")
        else:
            # Sort players by stack size for the final display (optional)
            final_players.sort(key=lambda p: p.money if hasattr(p, 'money') else 0, reverse=True)
            for p in final_players:
                p_id = p.id if hasattr(p, 'id') else 'N/A'
                money_val = p.money if hasattr(p, 'money') else 'N/A'
                money_str = f"${money_val}"
                is_ai_player = AI_ONLY_MODE or (hasattr(p, 'id') and p.id != HUMAN_PLAYER_ID)
                display_name = p.name if hasattr(p, 'name') else f"P{p_id}"
                # Consistent indicators
                type_indicator_log = " (AI)" if is_ai_player else " (Human)"
                type_indicator_term = colorize(" (AI)", Colors.MAGENTA) if is_ai_player else colorize(" (Human)", Colors.GREEN)
                stack_line = f"  - {display_name}{type_indicator_log} (ID:{p_id}): {money_str}"
                print(f"  - {colorize(display_name, Colors.CYAN)}{type_indicator_term}: {colorize(money_str, Colors.BRIGHT_GREEN)}")
                log_stacks.append(stack_line)

        ai_logger.info("\n".join(log_stacks))
        print(Colors.RESET) # Reset terminal colors at the very end