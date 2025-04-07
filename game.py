# -------- START: Code with Custom Player Names --------

import sys
import time      # For sleep
import traceback # For better error reporting
import os        # For screen clearing and dotenv
import requests  # For API calls
import random    # To assign models randomly and shuffle players
import logging   # For logging AI interactions
from collections import deque, defaultdict # For message history
from dotenv import load_dotenv # To load API key

# Necessary imports from pokerlib
from pokerlib import Player, PlayerSeats, Table
from pokerlib.enums import ( Hand, Rank, RoundPrivateOutId, RoundPublicInId,
                            RoundPublicOutId, Suit, TablePrivateOutId,
                            TablePublicInId, TablePublicOutId, Turn )

# --- Configuration ---

# --- ANSI Color Codes ---
class Colors:
    RESET = "\033[0m"; BOLD = "\033[1m"; UNDERLINE = "\033[4m"; BLACK = "\033[30m"; RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"; BRIGHT_BLACK = "\033[90m"; BRIGHT_RED = "\033[91m"; BRIGHT_GREEN = "\033[92m"; BRIGHT_YELLOW = "\033[93m"; BRIGHT_BLUE = "\033[94m"; BRIGHT_MAGENTA = "\033[95m"; BRIGHT_CYAN = "\033[96m"; BRIGHT_WHITE = "\033[97m"; BG_BLACK = "\033[40m"; BG_RED = "\033[41m"; BG_GREEN = "\033[42m"; BG_YELLOW = "\033[43m"; BG_BLUE = "\033[44m"; BG_MAGENTA = "\033[45m"; BG_CYAN = "\033[46m"; BG_WHITE = "\033[47m"
def colorize(text, color): return f"{color}{text}{Colors.RESET}"

RANK_MAP = { Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5", Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9", Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A" }
SUIT_MAP = {Suit.SPADE: "♠", Suit.CLUB: "♣", Suit.DIAMOND: "♦", Suit.HEART: "♥"}
SUIT_COLOR_MAP = { Suit.SPADE: Colors.WHITE, Suit.CLUB: Colors.WHITE, Suit.DIAMOND: Colors.BRIGHT_RED, Suit.HEART: Colors.BRIGHT_RED }

VERBOSE = False
ALWAYS_CONTINUE = False
AI_ONLY_MODE = False # <<< SET TO True FOR ALL AI, False FOR HUMAN vs AI >>>
HUMAN_PLAYER_ID = 1 # ID of the human player if AI_ONLY_MODE is False
HUMAN_PLAYER_NAME = "Aum" # <<< Name for the human player >>>
CLEAR_SCREEN = True
AI_MODEL_LIST = [ # Models for AI opponents
    'mistralai/Mistral-7B-Instruct-v0.3',
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'upstage/SOLAR-10.7B-Instruct-v1.0',
    'Gryphe/MythoMax-L2-13b',
    # 'mistralai/Mistral-Small-24B-Instruct-2501',
    # 'google/gemma-2-27b-it',
    # 'Qwen/QwQ-32B-Preview',
    # 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    # 'Qwen/Qwen2.5-72B-Instruct-Turbo',
    # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    # 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    # 'microsoft/WizardLM-2-8x22B',
    # 'databricks/dbrx-instruct',
    # 'deepseek-ai/DeepSeek-V3',
    # 'deepseek-ai/DeepSeek-R1',
]
# Dictionary mapping full model names to desired short names for display
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

NUM_AI_MODELS = len(AI_MODEL_LIST)
NUM_PLAYERS = NUM_AI_MODELS if AI_ONLY_MODE else NUM_AI_MODELS + 1
if NUM_PLAYERS < 2: print("Error: Need at least 2 total players."); sys.exit(1)
if NUM_AI_MODELS == 0 and AI_ONLY_MODE: print("Error: AI_ONLY_MODE=True but AI_MODEL_LIST empty."); sys.exit(1)

AI_TEMPERATURE = 0.7; AI_REQUEST_TIMEOUT = 60; AI_RETRY_DELAY = 7
LOG_FILE_NAME = 'ai_poker_log.txt'

# --- Logging Setup ---
if os.path.exists(LOG_FILE_NAME): os.remove(LOG_FILE_NAME)
ai_logger = logging.getLogger('AIPokerLog'); ai_logger.setLevel(logging.INFO); ai_logger.propagate = False
file_handler = logging.FileHandler(LOG_FILE_NAME); file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter); ai_logger.addHandler(file_handler)
ai_logger.info("--- AI Poker Log Initialized ---")

load_dotenv(); TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
if (AI_ONLY_MODE or NUM_AI_MODELS > 0) and not TOGETHER_AI_API_KEY: error_msg = "ERROR: AI players enabled, TOGETHER_AI_API_KEY missing."; print(error_msg); ai_logger.error(error_msg); sys.exit(1)
elif (AI_ONLY_MODE or NUM_AI_MODELS > 0): ai_logger.info("TOGETHER_AI_API_KEY loaded.")



# --- Helper Functions ---
def format_card_terminal(card):
    """Formats a card tuple (Rank, Suit) into a short, colored string."""
    # Default return value
    fallback_card = colorize("??", Colors.BRIGHT_BLACK)

    # Check validity first
    if not card or not hasattr(card, '__len__') or len(card) != 2:
        return fallback_card

    try:
        rank, suit = card
        # Use .get with default to prevent KeyError if somehow invalid enum sneaks through
        rank_str = RANK_MAP.get(rank)
        suit_str = SUIT_MAP.get(suit)
        # If rank or suit was invalid, return fallback
        if rank_str is None or suit_str is None:
            return fallback_card

        color = SUIT_COLOR_MAP.get(suit, Colors.WHITE)
        card_text = f"{rank_str}{suit_str}"
        if rank >= Rank.JACK: card_text = Colors.BOLD + card_text
        return colorize(card_text, color) # Return formatted string

    except (TypeError, ValueError, IndexError) as e: # Reduced exceptions caught
        # print(f"Error formatting card {card}: {e}") # Optional debug
        return fallback_card # Fallback return on error
def format_cards_terminal(cards):
    if hasattr(cards, '__iter__'): return " ".join(format_card_terminal(c) for c in cards); return ""
def format_card_for_ai(card): # ( Remains Correct )
    if not card: return "??";
    try:
        if hasattr(card, '__len__') and len(card) == 2: rank, suit = card; return f"{RANK_MAP[rank]}{SUIT_MAP[suit]}"
        else: return "??"
    except (TypeError, KeyError, ValueError, IndexError): return "??"
def format_cards_for_ai(cards):
    if hasattr(cards, '__iter__'): return " ".join(format_card_for_ai(c) for c in cards); return ""
def format_hand_enum(hand_enum): return hand_enum.name.replace("_", " ").title() if hand_enum else "Unknown Hand"
def clear_terminal():
    if CLEAR_SCREEN: os.system('cls' if os.name == 'nt' else 'clear')

# --- AI Interaction Functions ---
def query_together_ai(model_name, messages, temperature): # ( Remains the same )
    api_endpoint = "https://api.together.xyz/v1/chat/completions"; headers = { "Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json", "User-Agent": "PokerCLI-AI-Test" }
    payload = { "model": model_name, "messages": messages, "temperature": temperature, "max_tokens": 150 }; max_retries = 3; attempt = 0
    while attempt < max_retries:
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=AI_REQUEST_TIMEOUT); response.raise_for_status(); data = response.json()
            if data and 'choices' in data and len(data['choices']) > 0 and 'message' in data['choices'][0] and 'content' in data['choices'][0]['message']: return data['choices'][0]['message']['content'].strip()
            else: warn_msg = f"W: Bad API response from {model_name}: {data}"; print(colorize(warn_msg, Colors.YELLOW)); ai_logger.warning(warn_msg); return None
        except requests.exceptions.Timeout:
            warn_msg = f"W: API request to {model_name} timed out."; print(colorize(warn_msg, Colors.YELLOW)); ai_logger.warning(warn_msg); attempt += 1
            if attempt >= max_retries: return None; print(colorize(f"Retrying in {AI_RETRY_DELAY}s...", Colors.YELLOW)); time.sleep(AI_RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            err_msg = f"Error querying {model_name}: {e}";
            if hasattr(e, 'response') and e.response is not None: err_msg += f" | Response: {e.response.text}"
            print(colorize(err_msg, Colors.RED)); ai_logger.error(err_msg)
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                attempt += 1; retry_delay = AI_RETRY_DELAY * (attempt + 1); print(colorize(f"Rate limit. Retrying in {retry_delay}s...", Colors.YELLOW)); time.sleep(retry_delay)
            else: return None
        except Exception as e: err_msg = f"Unexpected API error {model_name}: {e}"; print(colorize(err_msg, Colors.RED)); ai_logger.error(err_msg); return None
    fail_msg = f"API call to {model_name} failed after {max_retries} retries."; print(colorize(fail_msg, Colors.RED)); ai_logger.error(fail_msg); return None

def parse_ai_action(response_text): # ( Remains the same )
    if not response_text: return RoundPublicInId.CHECK, {}
    response_lines = response_text.lower().split('\n'); action = None; raise_by = 0; action_line_found = False
    for line in response_lines:
        line = line.strip()
        if line.startswith("fold"): action = RoundPublicInId.FOLD; action_line_found = True; break
        if line.startswith("check"): action = RoundPublicInId.CHECK; action_line_found = True; break
        if line.startswith("call"): action = RoundPublicInId.CALL; action_line_found = True; break
        if line.startswith("raise"):
            action = RoundPublicInId.RAISE; action_line_found = True; parts = line.split()
            for i, part in enumerate(parts):
                 if part == "amount" and i+1 < len(parts) and parts[i+1].strip(':').isdigit(): raise_by = int(parts[i+1].strip(':')); break
                 elif part == "by" and i+1 < len(parts) and parts[i+1].isdigit(): raise_by = int(parts[i+1]); break
                 elif part.strip(':').isdigit(): raise_by = int(part.strip(':')); break
            break
    if not action_line_found:
         for line in response_lines:
            line = line.strip();
            if line.startswith("action:"):
                line_content = line[len("action:"):].strip()
                if "raise" in line_content:
                    action = RoundPublicInId.RAISE; parts = line_content.split()
                    for i, part in enumerate(parts):
                        if part.isdigit(): raise_by = int(part); break
                        elif part in ["by", "amount"] and i+1 < len(parts) and parts[i+1].strip(':').isdigit(): raise_by = int(parts[i+1].strip(':')); break
                    break
                elif "call" in line_content: action = RoundPublicInId.CALL; break
                elif "check" in line_content: action = RoundPublicInId.CHECK; break
                elif "fold" in line_content: action = RoundPublicInId.FOLD; break
    if action == RoundPublicInId.RAISE and raise_by <= 0: print(colorize(f"[AI WARN] RAISE w/o amount. Default CALL.", Colors.YELLOW)); action = RoundPublicInId.CALL; raise_by = 0
    elif action is None: print(colorize(f"[AI WARN] No action found: '{response_text}'. Default CHECK.", Colors.YELLOW)); action = RoundPublicInId.CHECK; raise_by = 0
    kwargs = {'raise_by': raise_by} if action == RoundPublicInId.RAISE else {}; return action, kwargs

def format_state_for_ai(table_state, player_id_acting): # ( Remains the same as previous correct version)
    lines = [];
    if not table_state.round or not hasattr(table_state.round, 'turn'): return "Error: No round or turn."
    try:
        lines.append(f"## Poker Hand State - Round {table_state.round.id}"); lines.append(f"**Current Stage:** {table_state.round.turn.name}")
        board_cards = [tuple(c) for c in table_state.round.board if isinstance(c, (list,tuple)) and len(c)==2]
        lines.append(f"**Board:** [ {format_cards_for_ai(board_cards)} ] ({len(board_cards)} cards)")
        pot_total = sum(table_state.round.pot_size) if hasattr(table_state.round, 'pot_size') and isinstance(table_state.round.pot_size, list) else 0
        lines.append(f"**Total Pot:** ${pot_total}")
        lines.append("\n**Players:** (Order is position relative to dealer)"); acting_player_obj = None; max_name_len = 0
        players_in_round = table_state.round.players if hasattr(table_state.round, 'players') else []
        if players_in_round: max_name_len = max(len(p.name) for p in players_in_round if p and hasattr(p, 'name'))
        for idx, p in enumerate(players_in_round):
            if not p or not hasattr(p, 'id'): continue
            is_acting = (p.id == player_id_acting) # Correctly placed
            if is_acting: acting_player_obj = p
            name_str = (p.name if hasattr(p, 'name') else f'P{p.id}').ljust(max_name_len); money_str = f"${p.money if hasattr(p, 'money') else 0}".ljust(6); cards_str = "( ? ? )"
            if p.id in table_state._player_cards and is_acting: cards_str = f"( {format_cards_for_ai(table_state._player_cards[p.id])} )"
            status = [];
            if hasattr(p, 'is_folded') and p.is_folded: status.append("FOLDED");
            if hasattr(p, 'is_all_in') and p.is_all_in: status.append("ALL-IN")
            if hasattr(table_state.round, 'button') and idx == table_state.round.button: status.append("D")
            if hasattr(table_state.round, 'small_blind_player_index') and idx == table_state.round.small_blind_player_index: status.append("SB")
            if hasattr(table_state.round, 'big_blind_player_index') and idx == table_state.round.big_blind_player_index: status.append("BB")
            status_str = f"[{' '.join(status)}]" if status else ""; current_bet = 0
            if hasattr(p, 'turn_stake') and isinstance(p.turn_stake, list) and hasattr(table_state.round.turn, 'value') and len(p.turn_stake) > table_state.round.turn.value: current_bet = p.turn_stake[table_state.round.turn.value]
            bet_str = f"Bet:${current_bet}" if current_bet > 0 else ""; prefix = "**YOU ->**" if is_acting else "   -"
            lines.append(f"{prefix} {name_str} {money_str} {cards_str.ljust(10)} {bet_str.ljust(10)} {status_str}")
        lines.append("\n**Action Context:**"); current_round_bet = table_state.round.turn_stake if table_state.round and hasattr(table_state.round, 'turn_stake') else 0
        lines.append(f"- Current Bet to Match: ${current_round_bet}"); amount_to_call = table_state.to_call
        lines.append(f"- Amount You Need to Call: ${amount_to_call}"); lines.append(f"- Your Stack: ${acting_player_obj.money if acting_player_obj else 'N/A'}")
        possible_actions = ["FOLD"];
        if table_state.can_check: possible_actions.append("CHECK")
        elif amount_to_call > 0 and (acting_player_obj and acting_player_obj.money > 0): possible_actions.append(f"CALL({min(amount_to_call, acting_player_obj.money)})")
        if table_state.can_raise: possible_actions.append("RAISE")
        lines.append(f"- Possible Actions: {', '.join(possible_actions)}"); lines.append(f"\n**Task:** Respond ONLY with: {', '.join(['CALL' if 'CALL(' in a else a for a in possible_actions])}.");
        lines.append("If RAISING, add ' AMOUNT: X' on the same line, where X is the integer amount to raise BY. Example: 'RAISE AMOUNT: 20'")
        return "\n".join(lines)
    except Exception as e: ai_logger.error(f"Error during format_state_for_ai: {e}", exc_info=True); return f"Error: Could not format game state - {e}"

# --- Custom Table Class ---
class CommandLineTable(Table):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._player_cards = {}; self._current_action_player_id = None; self.round_over_flag = False
        self.ai_message_history = defaultdict(lambda: defaultdict(list)); self.ai_model_assignments = {}
        self.to_call = 0; self.can_check = False; self.can_raise = False; self.min_raise = 0

    # --- Modified assign_ai_models ---
    def assign_ai_models(self):
        ai_player_ids = []; player_group = self.seats.getPlayerGroup()
        for p in player_group:
             if AI_ONLY_MODE or p.id != HUMAN_PLAYER_ID: ai_player_ids.append(p.id)
        num_ai = len(ai_player_ids);
        if num_ai == 0 or not AI_MODEL_LIST: return;
        log_msg_assign = ["AI Model Assignments:"]
        self.ai_model_assignments.clear()
        for i, pid in enumerate(ai_player_ids):
            model_full_name = AI_MODEL_LIST[i % len(AI_MODEL_LIST)] # Get the full name
            self.ai_model_assignments[pid] = model_full_name # Store the full name
            # Use short name for display/log if available, else fallback
            model_short_name = AI_MODEL_SHORT_NAMES.get(model_full_name, model_full_name.split('/')[-1])
            assign_text = f"AI Player {self._get_player_name(pid)} assigned model: {model_short_name} ({model_full_name})"
            print(colorize(f"[INFO] {assign_text}", Colors.MAGENTA))
            log_msg_assign.append(f"  - Player {pid} ({self._get_player_name(pid)}): {model_full_name} (as {model_short_name})")
        if log_msg_assign: ai_logger.info("\n".join(log_msg_assign))
    # --- END MODIFICATION ---

    def _get_player_name(self, player_id): player = self.seats.getPlayerById(player_id); return player.name if player else f"P{player_id}"
    def _newRound(self, round_id):
        ai_logger.info(f"\n{'='*15} ROUND {round_id} START {'='*15}"); current_players = self.seats.getPlayerGroup();
        if not current_players: print(colorize("Error: No players.", Colors.RED)); self.round = None; return
        for player in current_players:
            if hasattr(player, "resetState"): player.resetState()
        self._player_cards.clear();
        if round_id in self.ai_message_history: self.ai_message_history[round_id].clear()
        # Assign models only if AI players exist and assignments haven't been made
        num_ai_needed = len([p for p in current_players if AI_ONLY_MODE or p.id != HUMAN_PLAYER_ID])
        if num_ai_needed > 0 and not self.ai_model_assignments : self.assign_ai_models()
        try: self.round = self.RoundClass( round_id, current_players, self.button, self.small_blind, self.big_blind ); self.round_over_flag = False
        except Exception as e: print(colorize(f"--- ROUND INIT ERROR (Round {round_id}) ---", Colors.RED)); traceback.print_exc(); self.round = None
    def _display_game_state(self):
        clear_terminal(); title = colorize("====== POKER GAME STATE ======", Colors.BRIGHT_CYAN + Colors.BOLD)
        separator = colorize("--------------------------------------------------", Colors.BRIGHT_BLACK); print(f"\n{title}")
        if ( not self.round or not hasattr(self.round, 'id') or not hasattr(self.round, 'players') or not isinstance(self.round.players, list) or not hasattr(self.round, 'turn') or not hasattr(self.round, 'board') or not hasattr(self.round, 'pot_size') or not isinstance(self.round.pot_size, list)):
            print(colorize("No active round or state missing.", Colors.YELLOW)); print(colorize("\nPlayers at table:", Colors.YELLOW))
            for player in self.seats.getPlayerGroup():
                money_str = f"${player.money}" if hasattr(player, 'money') else colorize("N/A", Colors.BRIGHT_BLACK)
                is_ai = AI_ONLY_MODE or player.id != HUMAN_PLAYER_ID
                display_name = player.name if hasattr(player, 'name') else f"P{player.id}"
                type_indicator = colorize(" (AI)", Colors.MAGENTA) if is_ai else colorize(" (Human)", Colors.GREEN)
                print(f"  - {colorize(display_name, Colors.CYAN)}{type_indicator}: {colorize(money_str, Colors.BRIGHT_GREEN)}")
            print(separator); return
        try:
            round_id = self.round.id; turn_name = self.round.turn.name; board_cards_list = [tuple(c) for c in self.round.board if isinstance(c, (list,tuple)) and len(c)==2]
            board_cards_str = format_cards_terminal(board_cards_list); pot_total = sum(self.round.pot_size); current_turn_enum = self.round.turn
            players_in_round = self.round.players; button_idx = self.round.button if hasattr(self.round, "button") else -1
            sb_idx = self.round.small_blind_player_index if hasattr(self.round, "small_blind_player_index") else -1; bb_idx = self.round.big_blind_player_index if hasattr(self.round, "big_blind_player_index") else -1
        except (AttributeError, TypeError, IndexError) as e: print(colorize(f"Error accessing round details: {e}", Colors.RED)); return
        print(f"Round: {colorize(str(round_id), Colors.WHITE)}   Turn: {colorize(turn_name, Colors.WHITE + Colors.BOLD)}"); print(f"Board: [ {board_cards_str} ]"); print(f"Pot:   {colorize(f'${pot_total}', Colors.BRIGHT_YELLOW + Colors.BOLD)}"); print(colorize("\nPlayers:", Colors.YELLOW))
        max_name_len = 0;
        if players_in_round: max_name_len = max(len(p.name) for p in players_in_round if p and hasattr(p, 'name'))
        for idx, player in enumerate(players_in_round):
            if not player or not hasattr(player, "id"): continue
            is_acting = (player.id == self._current_action_player_id); line_prefix = colorize(" > ", Colors.BRIGHT_YELLOW + Colors.BOLD) if is_acting else "   "
            player_name_str = (player.name if hasattr(player, 'name') else f'P{player.id}').ljust(max_name_len); player_name_colored = colorize(player_name_str, Colors.CYAN + (Colors.BOLD if is_acting else ""))
            money_val = player.money if hasattr(player, 'money') else 0; money_str = colorize(f"${money_val}", Colors.BRIGHT_GREEN); cards_str = colorize("( ? ? )", Colors.BRIGHT_BLACK)
            if player.id in self._player_cards: cards_str = f"( {format_cards_terminal(self._player_cards[player.id])} )"
            elif (self.round_over_flag or (hasattr(self.round, 'finished') and self.round.finished)) and not player.is_folded and hasattr(player, "cards"):
                 cards_data = player.cards;
                 if isinstance(cards_data, (list, tuple)):
                     cards_tuple_list = [tuple(c) for c in cards_data if isinstance(c, (list, tuple)) and len(c)==2]
                     if len(cards_tuple_list) == 2: cards_str = f"( {format_cards_terminal(cards_tuple_list)} )"
            status = []; player_round_index = idx
            if hasattr(player, 'is_folded') and player.is_folded: status.append(colorize("FOLDED", Colors.BRIGHT_BLACK))
            if hasattr(player, 'is_all_in') and player.is_all_in: status.append(colorize("ALL-IN", Colors.BRIGHT_RED + Colors.BOLD))
            if player_round_index == button_idx: status.append(colorize("D", Colors.WHITE + Colors.BOLD))
            if player_round_index == sb_idx: status.append(colorize("SB", Colors.YELLOW));
            if player_round_index == bb_idx: status.append(colorize("BB", Colors.YELLOW));
            status_str = " ".join(status); turn_stake_val = 0
            if ( hasattr(player, "turn_stake") and isinstance(player.turn_stake, list) and hasattr(current_turn_enum, 'value') and len(player.turn_stake) > current_turn_enum.value ): turn_stake_val = player.turn_stake[current_turn_enum.value]
            stake_str = colorize(f"[Bet: ${turn_stake_val}]", Colors.MAGENTA) if turn_stake_val > 0 else ""
            # Display Type (Human/AI) based on mode and ID
            type_indicator = ""
            is_ai_player = AI_ONLY_MODE or player.id != HUMAN_PLAYER_ID
            if not is_ai_player: type_indicator = colorize(" (Human)", Colors.GREEN)
            print(f"{line_prefix}{player_name_colored}{type_indicator} {money_str.ljust(8)} {cards_str.ljust(20)} {stake_str.ljust(15)} {status_str}")
        print(separator)
    def publicOut(self, out_id, **kwargs): # ( Remains the same )
        player_id = kwargs.get("player_id"); player_name_raw = self._get_player_name(player_id) if player_id else "System"
        player_name = colorize(player_name_raw, Colors.CYAN); msg = ""; prefix = ""; processed = False; self._current_action_player_id = None
        self.can_check = False; self.can_raise = False; self.to_call = 0; self.min_raise = self.big_blind
        is_round_event = isinstance(out_id, RoundPublicOutId); is_table_event = isinstance(out_id, TablePublicOutId)
        if is_round_event:
            processed = True; prefix = colorize("[ROUND]", Colors.BLUE); player = self.seats.getPlayerById(player_id) if player_id else None
            if out_id == RoundPublicOutId.NEWROUND: msg = "Dealing cards..."
            elif out_id == RoundPublicOutId.NEWTURN: prefix = ""; self._display_game_state()
            elif out_id == RoundPublicOutId.SMALLBLIND: msg = f"{player_name} posts {colorize('SB', Colors.YELLOW)} ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.BIGBLIND: msg = f"{player_name} posts {colorize('BB', Colors.YELLOW)} ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.PLAYERCHECK: prefix = colorize("[ACTION]", Colors.GREEN); msg = f"{player_name} checks"
            elif out_id == RoundPublicOutId.PLAYERCALL: prefix = colorize("[ACTION]", Colors.GREEN); msg = f"{player_name} calls ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.PLAYERFOLD: prefix = colorize("[ACTION]", Colors.BRIGHT_BLACK); msg = f"{player_name} folds"
            elif out_id == RoundPublicOutId.PLAYERRAISE: prefix = colorize("[ACTION]", Colors.BRIGHT_MAGENTA); msg = f"{player_name} raises by ${kwargs['raised_by']} (bets ${kwargs['paid_amount']})"
            elif out_id == RoundPublicOutId.PLAYERISALLIN: prefix = colorize("[INFO]", Colors.BRIGHT_RED); msg = f"{player_name} is {colorize('ALL-IN!', Colors.BOLD)}"
            elif out_id == RoundPublicOutId.PLAYERWENTALLIN: prefix = colorize("[ACTION]", Colors.BRIGHT_RED + Colors.BOLD); msg = f"{player_name} goes ALL-IN with ${kwargs['paid_amount']}!"
            elif out_id == RoundPublicOutId.PLAYERACTIONREQUIRED:
                prefix = ""; self._current_action_player_id = player_id; self.to_call = kwargs.get('to_call', 0)
                self.can_check = self.to_call == 0;
                if player: self.can_raise = player.money > self.to_call
                else: self.can_raise = False
                self.min_raise = self.big_blind
            elif out_id == RoundPublicOutId.PLAYERCHOICEREQUIRED: pass # Ignoring
            elif out_id == RoundPublicOutId.PUBLICCARDSHOW:
                prefix = colorize("[SHOWDOWN]", Colors.WHITE); shown_cards_raw = kwargs.get('cards', []);
                shown_cards_tuples = [tuple(c) for c in shown_cards_raw if isinstance(c, (list, tuple)) and len(c)==2]
                if player and shown_cards_tuples: self._player_cards[player.id] = tuple(shown_cards_tuples) # Store shown cards
                hand_name = format_hand_enum(kwargs.get('handenum'))
                msg = f"{player_name} shows {format_cards_terminal(shown_cards_tuples)} ({hand_name})" # Use terminal formatter
            elif out_id == RoundPublicOutId.DECLAREPREMATUREWINNER: prefix = colorize("[WINNER]", Colors.BRIGHT_YELLOW + Colors.BOLD); msg = f"{player_name} wins ${kwargs['money_won']} (Premature)"
            elif out_id == RoundPublicOutId.DECLAREFINISHEDWINNER: prefix = colorize("[WINNER]", Colors.BRIGHT_YELLOW + Colors.BOLD); hand_name = format_hand_enum(kwargs.get('handname')); msg = f"{player_name} wins ${kwargs['money_won']} with {hand_name}"
            elif out_id == RoundPublicOutId.ROUNDFINISHED: prefix = ""; msg = colorize("\n======= ROUND FINISHED =======", Colors.BRIGHT_CYAN); self.round_over_flag = True; self._display_game_state()
            elif out_id == RoundPublicOutId.ROUNDCLOSED: prefix = colorize("[Internal]", Colors.BRIGHT_BLACK); msg = "Round Closed State."; self._player_cards.clear()
        elif is_table_event:
            processed = True; prefix = colorize("[TABLE]", Colors.MAGENTA)
            if out_id == TablePublicOutId.PLAYERJOINED: msg = f"{player_name} joined seat {kwargs['player_seat']}"
            elif out_id == TablePublicOutId.PLAYERREMOVED:
                 msg = f"{player_name} left table";
                 if player_id in self._player_cards: del self._player_cards[player_id]
                 if player_id in self.ai_model_assignments: del self.ai_model_assignments[player_id]
            elif out_id == TablePublicOutId.NEWROUNDSTARTED: prefix = ""; msg = ""
            elif out_id == TablePublicOutId.ROUNDNOTINITIALIZED: prefix = colorize("[ERROR]", Colors.RED); msg = "No round running"
            elif out_id == TablePublicOutId.ROUNDINPROGRESS: prefix = colorize("[ERROR]", Colors.RED); msg = "Round already in progress"
            elif out_id == TablePublicOutId.INCORRECTNUMBEROFPLAYERS: prefix = colorize("[ERROR]", Colors.RED); msg = "Need 2+ players"
        should_print = bool(msg)
        if is_round_event and out_id in [RoundPublicOutId.NEWTURN, RoundPublicOutId.ROUNDFINISHED, RoundPublicOutId.PLAYERACTIONREQUIRED]: should_print = False
        if is_table_event and out_id == TablePublicOutId.NEWROUNDSTARTED: should_print = False
        if should_print: print(f"{prefix} {msg}")
        elif not processed and out_id != RoundPublicOutId.PLAYERCHOICEREQUIRED: print(colorize(f"Unhandled Out: ID={out_id} Data: {kwargs}", Colors.BRIGHT_BLACK))
    def privateOut(self, player_id, out_id, **kwargs): # ( Remains the same )
        player_name_raw = self._get_player_name(player_id); player_name = colorize(player_name_raw, Colors.CYAN)
        prefix = colorize(f"[PRIVATE to {player_name_raw}]", Colors.YELLOW); msg = ""
        if out_id == RoundPrivateOutId.DEALTCARDS:
            cards_raw = kwargs.get('cards');
            if cards_raw and len(cards_raw) == 2:
                cards_tuples = tuple(tuple(c) for c in cards_raw if isinstance(c, (list, tuple)) and len(c)==2)
                if len(cards_tuples) == 2: self._player_cards[player_id] = cards_tuples; msg = f"You are dealt {format_cards_terminal(cards_tuples)}" # Use terminal formatter
                else: msg = colorize("Card conversion error.", Colors.RED)
            else: msg = colorize("Dealing error (no cards data).", Colors.RED)
        elif out_id == TablePrivateOutId.BUYINTOOLOW: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Buy-in of ${self.buyin} required."
        elif out_id == TablePrivateOutId.TABLEFULL: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Table full"
        elif out_id == TablePrivateOutId.PLAYERALREADYATTABLE: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Already seated"
        elif out_id == TablePrivateOutId.PLAYERNOTATTABLE: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Not at table"
        elif out_id == TablePrivateOutId.INCORRECTSEATINDEX: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Bad seat index"
        else: prefix = colorize(f"[UNHANDLED PRIVATE to {player_name_raw}]", Colors.BRIGHT_BLACK); msg = f"ID={out_id} Data: {kwargs}"
        # Only print private message if it's for the human player in mixed mode
        if msg and not AI_ONLY_MODE and player_id == HUMAN_PLAYER_ID: print(f"{prefix} {msg}")

# --- Main Game Logic (Human Action Prompt) ---
def get_player_action(player_name, to_call, player_money, can_check, can_raise): # ( Remains the same )
    prompt_header = colorize(f"--- Your Turn ({player_name}) ---", Colors.BRIGHT_YELLOW + Colors.BOLD); print(prompt_header)
    actions = ["FOLD"]; action_color_map = { "FOLD": Colors.BRIGHT_BLACK, "CHECK": Colors.BRIGHT_GREEN, "CALL": Colors.BRIGHT_CYAN, "RAISE": Colors.BRIGHT_MAGENTA }
    action_parts = [colorize("FOLD", action_color_map["FOLD"])]
    if can_check: actions.append("CHECK"); action_parts.append(colorize("CHECK", action_color_map["CHECK"]))
    elif to_call > 0 and player_money > 0: actions.append("CALL"); effective_call = min(to_call, player_money); action_parts.append(colorize(f"CALL({effective_call})", action_color_map["CALL"]))
    if can_raise: actions.append("RAISE"); action_parts.append(colorize("RAISE", action_color_map["RAISE"]))
    print(f"Available actions: {' / '.join(action_parts)}")
    while True:
        try: action_str = input(colorize("Enter action: ", Colors.WHITE)).upper().strip()
        except EOFError: print(colorize("\nInput ended.", Colors.RED)); return RoundPublicInId.FOLD, {}
        if action_str == "CALL" and can_check: print(colorize("No bet to call. Use CHECK.", Colors.YELLOW)); continue
        if action_str == "CHECK" and not can_check: print(colorize(f"Cannot check. Bet is ${to_call}. Use CALL or FOLD.", Colors.YELLOW)); continue
        if action_str not in actions: print(colorize("Invalid action.", Colors.RED) + f" Choose from: {', '.join(actions)}"); continue
        if action_str == "RAISE":
            if not can_raise: print(colorize("Error: Raise not available.", Colors.RED)); continue
            min_raise = table.big_blind; max_raise = player_money - to_call
            if max_raise < min_raise and player_money > to_call: min_raise = max_raise
            if max_raise <= 0: print(colorize("Cannot raise, not enough funds.", Colors.RED)); continue
            while True:
                try:
                    prompt_range = ( f"(min {min_raise}, max {max_raise})" if min_raise < max_raise else f"(exactly {max_raise} to go all-in)" )
                    raise_by_str = input(colorize(f"  Raise BY how much? {prompt_range}: ", Colors.WHITE))
                    if not raise_by_str.isdigit(): raise ValueError("Input not a digit")
                    raise_by = int(raise_by_str); is_all_in_raise = to_call + raise_by >= player_money
                    if raise_by <= 0: print(colorize("Raise amount must be positive.", Colors.YELLOW))
                    elif raise_by > max_raise: print(colorize(f"Max raise BY is {max_raise}.", Colors.YELLOW))
                    elif raise_by < min_raise and not is_all_in_raise: print(colorize(f"Minimum raise BY is {min_raise} (unless all-in).", Colors.YELLOW))
                    else: return RoundPublicInId.RAISE, {"raise_by": min(raise_by, max_raise)}
                except ValueError: print(colorize("Invalid amount. Please enter a number.", Colors.YELLOW))
                except EOFError: print(colorize("\nInput ended.", Colors.RED)); return RoundPublicInId.FOLD, {}
        elif action_str == "FOLD": return RoundPublicInId.FOLD, {}
        elif action_str == "CHECK":
            if can_check: return RoundPublicInId.CHECK, {}
        elif action_str == "CALL":
            if not can_check: return RoundPublicInId.CALL, {}
        print(colorize("Error processing action. Please try again.", Colors.RED))

# --- AI Action Logic (With Logging) ---
def get_ai_action(table_state: CommandLineTable, player_id): # ( Remains the same )
    player_name = table_state._get_player_name(player_id); model_name = table_state.ai_model_assignments.get(player_id)
    if not model_name: print(colorize(f"E: No model for AI {player_name}", Colors.RED)); ai_logger.error(f"No model assigned to AI {player_id}."); return RoundPublicInId.FOLD, {}
    if VERBOSE:
        print(colorize(f"--- AI Thinking ({player_name} using {model_name.split('/')[-1]}) ---", Colors.MAGENTA))
    time.sleep(0.5)
    prompt = format_state_for_ai(table_state, player_id); # Uses fixed function now
    if "Error:" in prompt: print(colorize(f"E: formatting state for AI {player_name}: {prompt}", Colors.RED)); ai_logger.error(f"Error formatting prompt for AI {player_id}: {prompt}"); return RoundPublicInId.FOLD, {}
    round_id = table_state.round.id; history = table_state.ai_message_history[round_id][player_id]
    system_prompt = { "role": "system", "content": "You are a Texas Hold'em poker AI. Analyze the game state and decide your next action. Focus on making a reasonable poker play. Respond ONLY with the action name (FOLD, CHECK, CALL, RAISE). If raising, add ' AMOUNT: X' on the same line, where X is the integer amount to raise BY. Do not add any other explanation or text." }
    messages = [system_prompt] + history + [{"role": "user", "content": prompt}]
    ai_logger.info(f"--- AI Turn: {player_name} (Model: {model_name}, Round: {round_id}) ---"); ai_logger.info(f"Prompt Sent:\n{'-'*20}\n{prompt}\n{'-'*20}")
    ai_response_text = query_together_ai(model_name, messages, AI_TEMPERATURE)
    ai_logger.info(f"Raw Response:\n{'-'*20}\n{ai_response_text or '<< No Response >>'}\n{'-'*20}")
    if ai_response_text:
        if VERBOSE:
            print(colorize(f"AI Raw Response ({player_name}): ", Colors.BRIGHT_BLACK) + f"{ai_response_text}")
        action_enum, action_kwargs = parse_ai_action(ai_response_text)
        player = table_state.seats.getPlayerById(player_id); is_possible = False
        if action_enum == RoundPublicInId.FOLD: is_possible = True
        elif action_enum == RoundPublicInId.CHECK: is_possible = table_state.can_check
        elif action_enum == RoundPublicInId.CALL: is_possible = not table_state.can_check and table_state.to_call > 0 and player.money > 0
        elif action_enum == RoundPublicInId.RAISE:
            is_possible = table_state.can_raise
            if is_possible:
                raise_by = action_kwargs.get('raise_by', 0); max_raise = player.money - table_state.to_call; min_r = table_state.min_raise
                if max_raise < min_r and player.money > table_state.to_call: min_r = max_raise
                if raise_by <= 0 or raise_by > max_raise or (raise_by < min_r and (table_state.to_call + raise_by < player.money)):
                    warn_msg = f"[AI WARN] AI {player_name} invalid RAISE {raise_by} (min:{min_r}, max:{max_raise})."; print(colorize(warn_msg, Colors.YELLOW)); ai_logger.warning(warn_msg + " Fallback required.")
                    is_possible = False
                    if not table_state.can_check and table_state.to_call > 0 and player.money > 0: fallback_msg = "Fallback: CALL"; print(colorize(fallback_msg, Colors.YELLOW)); ai_logger.info(fallback_msg); action_enum = RoundPublicInId.CALL; action_kwargs = {}; is_possible = True
                    else: fallback_msg = "Fallback: FOLD"; print(colorize(fallback_msg, Colors.YELLOW)); ai_logger.info(fallback_msg); action_enum = RoundPublicInId.FOLD; action_kwargs = {}; is_possible = True
                else: action_kwargs['raise_by'] = min(raise_by, max_raise)
        if not is_possible:
             warn_msg = f"[AI WARN] AI {player_name} chose impossible {action_enum.name}."; print(colorize(warn_msg, Colors.YELLOW)); ai_logger.warning(warn_msg + " Fallback required.")
             if not table_state.can_check and table_state.to_call > 0 and player.money > 0: fallback_msg = "Fallback: CALL"; print(colorize(fallback_msg, Colors.YELLOW)); ai_logger.info(fallback_msg); action_enum = RoundPublicInId.CALL; action_kwargs = {}
             else: fallback_msg = "Fallback: FOLD"; print(colorize(fallback_msg, Colors.YELLOW)); ai_logger.info(fallback_msg); action_enum = RoundPublicInId.FOLD; action_kwargs = {}
        history.append({"role": "user", "content": prompt}); assistant_response_content = f"{action_enum.name}"
        if action_enum == RoundPublicInId.RAISE: assistant_response_content += f" AMOUNT: {action_kwargs['raise_by']}"
        history.append({"role": "assistant", "content": assistant_response_content})
        parsed_action_log = f"Validated Action: {action_enum.name} {action_kwargs}"
        if VERBOSE:
            print(colorize(f"AI Validated Action ({player_name}): {action_enum.name} {action_kwargs}", Colors.MAGENTA))
        ai_logger.info(parsed_action_log)
        return action_enum, action_kwargs
    else: fail_msg = f"AI ({player_name}) failed to respond. Defaulting to FOLD."; print(colorize(fail_msg, Colors.RED)); ai_logger.error(fail_msg); return RoundPublicInId.FOLD, {}

# --- Main Execution ---
if __name__ == "__main__":
    BUYIN = 1000; SMALL_BLIND = 5; BIG_BLIND = 10 # Adjusted Buyin
    NUM_AI_MODELS = len(AI_MODEL_LIST)
    NUM_PLAYERS = NUM_AI_MODELS if AI_ONLY_MODE else NUM_AI_MODELS + 1
    if NUM_PLAYERS < 2: print(colorize("Error: Need at least 2 total players.", Colors.RED)); sys.exit(1)

    table = CommandLineTable( _id=0, seats=PlayerSeats([None] * NUM_PLAYERS), buyin=BUYIN, small_blind=SMALL_BLIND, big_blind=BIG_BLIND )

    players = []
    player_id_counter = 1
    # Create Human Player if needed
    if not AI_ONLY_MODE:
        human_player = Player(table_id=table.id, _id=HUMAN_PLAYER_ID, name=HUMAN_PLAYER_NAME, money=BUYIN)
        players.append(human_player)
        # Ensure counter starts correctly for AI IDs if human exists
        if HUMAN_PLAYER_ID >= player_id_counter: player_id_counter = HUMAN_PLAYER_ID + 1

    # Create AI Players
    ai_player_count = 0
    player_ids_generated = {HUMAN_PLAYER_ID} if not AI_ONLY_MODE else set() # Keep track of used IDs
    for i in range(NUM_PLAYERS): # Ensure exactly NUM_PLAYERS are created
        # Determine if the current slot is for human or AI
        is_for_human = not AI_ONLY_MODE and i == 0 # Assume human is first if not AI only (before shuffle)

        if is_for_human:
            # Human player already created and added
            continue
        else:
            # Create AI player, ensuring unique ID
            while player_id_counter in player_ids_generated:
                player_id_counter += 1

            ai_player_id = player_id_counter
            player_ids_generated.add(ai_player_id)

            # Assign model and get short name for the Player object
            # Use ai_player_count to cycle through models for AI players
            if ai_player_count < NUM_AI_MODELS: # Check if enough models defined
                 model_full_name = AI_MODEL_LIST[ai_player_count % len(AI_MODEL_LIST)]
                 ai_short_name = AI_MODEL_SHORT_NAMES.get(model_full_name, f"AI_{ai_player_id}")
            else: # Fallback if more AI slots than models (shouldn't happen with current logic)
                 ai_short_name = f"AI_{ai_player_id}"

            ai_player = Player(table_id=table.id, _id=ai_player_id, name=ai_short_name, money=BUYIN)
            players.append(ai_player)
            ai_player_count += 1
            player_id_counter += 1 # Move to next potential ID


    random.shuffle(players) # Shuffle seating order
    for p in players: table.publicIn(p.id, TablePublicInId.BUYIN, player=p)

    clear_terminal(); print(colorize("\n--- Welcome to NLPoker! ---", Colors.BRIGHT_CYAN + Colors.BOLD))
    print(f"{NUM_PLAYERS} players at the table ({NUM_AI_MODELS} AI, {0 if AI_ONLY_MODE else 1} Human).")
    print(f"Buy-in: {colorize(f'${BUYIN}', Colors.BRIGHT_GREEN)} each. Blinds: {colorize(f'${SMALL_BLIND}/${BIG_BLIND}', Colors.YELLOW)}")
    if AI_ONLY_MODE: print(colorize("Mode: ALL AI Players", Colors.MAGENTA))
    else: print(colorize(f"Mode: Human ({HUMAN_PLAYER_NAME}) vs AI", Colors.MAGENTA))
    ai_logger.info(f"Game Started. AI_ONLY_MODE: {AI_ONLY_MODE}. Human ID: {HUMAN_PLAYER_ID if not AI_ONLY_MODE else 'N/A'}. Num AI: {NUM_AI_MODELS}.")

    round_count = 0
    try:
        while True:
            active_players_obj = table.seats.getPlayerGroup()
            if len(active_players_obj) < 2: print(colorize("\nNot enough players. Game over.", Colors.YELLOW)); break
            round_count += 1
            initiator = active_players_obj[0] if active_players_obj else None
            if not initiator: print(colorize("Error: No players left.", Colors.RED)); break
            table.publicIn( initiator.id, TablePublicInId.STARTROUND, round_id=round_count )

            while table.round and not table.round_over_flag:
                action_player_id_to_process = table._current_action_player_id
                if action_player_id_to_process:
                    table._current_action_player_id = None # Clear flag
                    player = table.seats.getPlayerById(action_player_id_to_process)
                    if not player: print(colorize(f"W: Action for missing P{action_player_id_to_process}", Colors.YELLOW)); continue
                    current_player_obj = None
                    if table.round and hasattr(table.round, "current_player"):
                        try: current_player_obj = table.round.current_player
                        except Exception: pass
                    if current_player_obj and player.id == current_player_obj.id:
                        if not all(hasattr(player, a) for a in ['money','stake','turn_stake']) or not isinstance(player.turn_stake, list) or not (table.round and hasattr(table.round, 'to_call')):
                             print(colorize(f"W: State not ready for {player.name}.", Colors.YELLOW)); time.sleep(0.1); table._current_action_player_id = action_player_id_to_process; continue

                        is_ai_turn = AI_ONLY_MODE or player.id != HUMAN_PLAYER_ID
                        if is_ai_turn: action_enum, action_kwargs = get_ai_action(table, player.id); time.sleep(0.5)
                        else: action_enum, action_kwargs = get_player_action( player.name, table.to_call, player.money, table.can_check, table.can_raise )

                        table.publicIn(player.id, action_enum, **action_kwargs) # Send action
                    elif current_player_obj:
                        req_for = f"{player.name}({action_player_id_to_process})"; curr = f"{current_player_obj.name}({current_player_obj.id})"
                        print(colorize(f"W: Action req mismatch. Req={req_for}, Current={curr}", Colors.YELLOW))
                time.sleep(0.05)

            if table.round: table.round = None # Clear library's round object
            ai_logger.info(f"--- ROUND {round_count} END ---"); print(colorize("\nRound ended. Final stacks:", Colors.BRIGHT_WHITE)); final_players = table.seats.getPlayerGroup()
            if not final_players: print("  No players remaining.")
            else:
                 log_stacks = ["Final Stacks:"]
                 for p in final_players:
                     money_val = p.money if hasattr(p, 'money') else 'N/A';
                     is_ai_player = AI_ONLY_MODE or p.id != HUMAN_PLAYER_ID
                     display_name = p.name if hasattr(p, 'name') else f"P{p.id}" # Use actual name
                     type_indicator_log = " (AI)" if is_ai_player else " (Human)"
                     type_indicator_term = colorize(" (AI)", Colors.MAGENTA) if is_ai_player else colorize(" (Human)", Colors.GREEN)
                     stack_line = f"  - {display_name}{type_indicator_log}: ${money_val}" # Log plain string
                     print(f"  - {colorize(display_name, Colors.CYAN)}{type_indicator_term}: {colorize(f'${money_val}', Colors.BRIGHT_GREEN)}") # Print colored string
                     log_stacks.append(stack_line)
                 ai_logger.info("\n".join(log_stacks))

            if not ALWAYS_CONTINUE: # Use the new flag
                if not AI_ONLY_MODE: # Only ask human if playing
                    try: cont = input(colorize("\nPlay another round? (y/n): ", Colors.WHITE)).lower();
                    except EOFError: print(colorize("\nInput ended.", Colors.RED)); break
                    if cont != 'y': break
                else: # In AI Only mode, pause and continue
                    time.sleep(3); print("\nStarting next round automatically...")
            else: # If ALWAYS_CONTINUE is True
                time.sleep(1); print("\nStarting next round automatically...")


    except KeyboardInterrupt: print(colorize("\nCtrl+C detected. Exiting game.", Colors.YELLOW))
    except Exception as e: print(colorize("\n--- UNEXPECTED ERROR ---", Colors.RED+Colors.BOLD)); traceback.print_exc(); print(colorize("-----", Colors.RED+Colors.BOLD)); ai_logger.exception("UNEXPECTED ERROR")
    finally:
        game_end_msg = "\n--- Game Ended ---"; print(colorize(game_end_msg, Colors.BRIGHT_CYAN + Colors.BOLD)); ai_logger.info(game_end_msg)
        print(colorize("Final Stacks:", Colors.WHITE)); final_players = table.seats.getPlayerGroup()
        log_stacks = ["Final Stacks:"]
        if not final_players: print("  No players remaining."); log_stacks.append("  No players remaining.")
        else:
            for p in final_players:
                money_str = f"${p.money}" if hasattr(p, 'money') else "N/A";
                is_ai_player = AI_ONLY_MODE or p.id != HUMAN_PLAYER_ID
                display_name = p.name if hasattr(p, 'name') else f"P{p.id}" # Use actual name
                type_indicator_log = " (AI)" if is_ai_player else " (Human)"
                type_indicator_term = colorize(" (AI)", Colors.MAGENTA) if is_ai_player else colorize(" (Human)", Colors.GREEN)
                stack_line = f"  - {display_name}{type_indicator_log}: {money_str}" # Log plain string
                print(f"  - {colorize(display_name, Colors.CYAN)}{type_indicator_term}: {colorize(money_str, Colors.BRIGHT_GREEN)}") # Print colored string
                log_stacks.append(stack_line)
        ai_logger.info("\n".join(log_stacks)); print(Colors.RESET)

# -------- END: Code with Final Fixes and Dynamic Players --------