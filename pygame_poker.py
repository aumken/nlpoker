#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyGame implementation of Texas Hold'em Poker with retro/8-bit aesthetic.
Based on the original terminal-based poker game.
"""

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from enum import Enum, auto

import pygame
import pygame.freetype
import pygame.gfxdraw
import requests
from dotenv import load_dotenv
# Import the necessary libraries from the original game
from pokerlib import Player, PlayerSeats, Table
from pokerlib.enums import (Hand, Rank, RoundPrivateOutId, RoundPublicInId,
                            RoundPublicOutId, Suit, TablePrivateOutId,
                            TablePublicInId, TablePublicOutId, Turn)
from pygame.locals import *

# Import Deuces for hand evaluation
try:
    from deuces import Card, Deck, Evaluator
except ImportError:
    print("Error: 'deuces' library not found. Please install it: pip install deuces")
    sys.exit(1)

# --- Game States ---
class GameState(Enum):
    MAIN_MENU = auto()
    SETTINGS = auto()
    MODEL_SELECT = auto()
    PLAYING = auto()
    PAUSED = auto()
    ROUND_END = auto()

# Load environment variables
load_dotenv()
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")

# --- Default Game Settings ---
# These will be configurable in the settings menu
DEFAULT_SETTINGS = {
    "BUYIN": 1000,
    "SMALL_BLIND": 5,
    "BIG_BLIND": 10,
    "AI_ONLY_MODE": True,
    "HUMAN_PLAYER_NAME": "Player",
    "HUMAN_PLAYER_ID": 1,
    "SHOW_PROBABILITIES": True,
    "PROBABILITY_SIMULATIONS": 5000,
    "AI_TEMPERATURE": 1.0,
    "AI_REQUEST_TIMEOUT": 60,
    "AI_RETRY_DELAY": 5,
    "AI_THINK_DELAY": 0.5,  # Added setting for AI thinking delay (in seconds)
    "SOUND_EFFECTS": True,
    "MUSIC": True,
    "ANIMATION_SPEED": 5,  # 1-10 scale, higher is faster
}

# --- AI Models Configuration ---
DEFAULT_AI_MODELS = [
    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'Qwen/Qwen2.5-72B-Instruct-Turbo',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
]

DEFAULT_AI_MODEL_SHORT_NAMES = {
    'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral v0.3 7B',
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': 'Llama 3.1 8B',
    'upstage/SOLAR-10.7B-Instruct-v1.0': 'Upstage Solar 11B',
    'mistralai/Mistral-Small-24B-Instruct-2501': 'Mistral Small 3 24B',
    'google/gemma-2-27b-it': 'Gemma 2 27B',
    'Qwen/QwQ-32B-Preview': 'Qwen QwQ 32B',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo': 'Llama 3.3 70B',
    'Qwen/Qwen2.5-72B-Instruct-Turbo': 'Qwen 2.5 72B',
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 'Mixtral 8x7B',
    'microsoft/WizardLM-2-8x22B': 'WizardLM-2 8x22B',
    'databricks/dbrx-instruct': 'DBRX Instruct',
    'deepseek-ai/DeepSeek-V3': 'DeepSeek V3',
    'deepseek-ai/DeepSeek-R1': 'DeepSeek R1',
}

# --- Card Mappings (from original game) ---
RANK_MAP_POKERLIB = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5", Rank.SIX: "6",
    Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9", Rank.TEN: "T", Rank.JACK: "J",
    Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A"
}

SUIT_MAP_POKERLIB = { Suit.SPADE: "S", Suit.CLUB: "C", Suit.DIAMOND: "D", Suit.HEART: "H" }

SUIT_COLOR_MAP = {
    Suit.SPADE: (0, 0, 0),         # Black
    Suit.CLUB: (0, 0, 0),          # Black
    Suit.DIAMOND: (255, 100, 100), # Light Red
    Suit.HEART: (255, 100, 100)    # Light Red
}

# --- Constants ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 700  # Increased height
FPS = 60

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
POKER_GREEN = (53, 101, 77)
POKER_DARK_GREEN = (35, 75, 50)
GOLD = (255, 215, 0)

# AI Interaction Functions
def query_together_ai(model_name, messages, temperature, timeout=60, retry_delay=5):
    """Sends a prompt to the Together AI API and returns the model's response."""
    if not TOGETHER_AI_API_KEY:
        print("Error: TOGETHER_AI_API_KEY not set. Check your .env file.")
        return None
    
    api_endpoint = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "PokerPyGame-AI-Test"
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
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()
            
            # Check if the response structure is as expected
            if (data and 'choices' in data and len(data['choices']) > 0 and
                    'message' in data['choices'][0] and 'content' in data['choices'][0]['message']):
                return data['choices'][0]['message']['content'].strip()
            else:
                print(f"Warning: Unexpected API response structure from {model_name}")
                return None
        
        except requests.exceptions.Timeout:
            print(f"Warning: API request to {model_name} timed out (attempt {attempt + 1}/{max_retries}).")
            attempt += 1
            if attempt >= max_retries:
                return None
            print(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        
        except requests.exceptions.RequestException as e:
            print(f"Error querying {model_name}: {e}")
            return None
        
        except Exception as e:
            print(f"Unexpected error during API call to {model_name}: {e}")
            return None
    
    print(f"API call to {model_name} failed after {max_retries} retries.")
    return None

def parse_ai_action(response_text):
    """Parses the AI's text response to determine the poker action and raise amount."""
    if not response_text:
        return RoundPublicInId.CHECK, {}  # Default action if no response
    
    response_lines = response_text.lower().split('\n')
    action = None
    raise_by = 0
    
    # Primary parsing: Check for action keywords at the start of any line
    for line in response_lines:
        line = line.strip()
        if line.startswith("fold"):
            action = RoundPublicInId.FOLD
            break
        if line.startswith("check"):
            action = RoundPublicInId.CHECK
            break
        if line.startswith("call"):
            action = RoundPublicInId.CALL
            break
        if line.startswith("raise"):
            action = RoundPublicInId.RAISE
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "amount" and i + 1 < len(parts) and parts[i+1].strip(':').isdigit():
                    raise_by = int(parts[i+1].strip(':'))
                    break
                elif part == "by" and i + 1 < len(parts) and parts[i+1].isdigit():
                    raise_by = int(parts[i+1])
                    break
                elif i > 0 and parts[i-1] == "raise" and part.strip(':').isdigit():
                    raise_by = int(part.strip(':'))
                    break
                elif part.strip(':').isdigit():
                    raise_by = int(part.strip(':'))
            break
    
    # Secondary parsing: If no action found, look for "action:" prefix
    if not action:
        for line in response_lines:
            line = line.strip()
            if line.startswith("action:"):
                line_content = line[len("action:"):].strip()
                if "raise" in line_content:
                    action = RoundPublicInId.RAISE
                    parts = line_content.split()
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            raise_by = int(part)
                            break
                        elif part in ["by", "amount"] and i + 1 < len(parts) and parts[i+1].strip(':').isdigit():
                            raise_by = int(parts[i+1].strip(':'))
                            break
                    break
                elif "call" in line_content:
                    action = RoundPublicInId.CALL
                    break
                elif "check" in line_content:
                    action = RoundPublicInId.CHECK
                    break
                elif "fold" in line_content:
                    action = RoundPublicInId.FOLD
                    break
    
    # Validation and fallback
    if action == RoundPublicInId.RAISE and raise_by <= 0:
        print(f"Warning: AI responded RAISE but failed to specify a valid amount > 0. Defaulting to CALL.")
        action = RoundPublicInId.CALL
        raise_by = 0
    elif action is None:
        print(f"Warning: Could not parse action from AI response: '{response_text}'. Defaulting to CHECK.")
        action = RoundPublicInId.CHECK
        raise_by = 0
    
    # Prepare return values
    kwargs = {'raise_by': raise_by} if action == RoundPublicInId.RAISE else {}
    return action, kwargs

def format_state_for_ai(table_state, player_id_acting):
    """Formats the current game state into a markdown string suitable for the AI prompt."""
    lines = []
    
    # Basic state checks
    if not table_state.round or not hasattr(table_state.round, 'turn'):
        return "Error: Game state is missing round or turn information."
    if not hasattr(table_state, '_player_cards'):
        return "Error: Table state missing _player_cards information."
    
    try:
        # Round Information
        lines.append(f"## Poker Hand State - Round {table_state.round.id}")
        lines.append(f"**Current Stage:** {table_state.round.turn.name}")
        
        # Board Cards
        board_cards = []
        if hasattr(table_state.round, 'board'):
            board_cards = [tuple(c) for c in table_state.round.board if isinstance(c, (list,tuple)) and len(c)==2]
        
        # Format cards using the card notation from game.py
        formatted_board = []
        for card in board_cards:
            rank, suit = card
            rank_str = RANK_MAP_POKERLIB.get(rank, "?")
            suit_str = SUIT_MAP_POKERLIB.get(suit, "?")
            formatted_board.append(f"{rank_str}{suit_str}")
        
        board_str = " ".join(formatted_board) if formatted_board else ""
        lines.append(f"**Board:** [ {board_str} ] ({len(board_cards)} cards)")
        
        # Pot Size
        pot_total = 0
        if hasattr(table_state.round, 'pot_size') and isinstance(table_state.round.pot_size, list):
            pot_total = sum(table_state.round.pot_size)
        lines.append(f"**Total Pot:** ${pot_total}")
        
        # Player Information
        lines.append("\n**Players:** (Order is position relative to dealer)")
        acting_player_obj = None
        players_in_round = []
        
        if hasattr(table_state.round, 'players') and table_state.round.players:
            players_in_round = table_state.round.players
        
        for idx, p in enumerate(players_in_round):
            if not p or not hasattr(p, 'id'): continue
            
            is_acting = (p.id == player_id_acting)
            if is_acting:
                acting_player_obj = p
            
            # Player Name and Money
            name_str = p.name if hasattr(p, 'name') else f'P{p.id}'
            money_val = p.money if hasattr(p, 'money') else 0
            
            # Player Cards (Show only for the acting player)
            cards_str = "( ? ? )"  # Default hidden cards
            if is_acting and p.id in table_state._player_cards:
                player_cards = table_state._player_cards[p.id]
                formatted_cards = []
                for card in player_cards:
                    rank, suit = card
                    rank_str = RANK_MAP_POKERLIB.get(rank, "?")
                    suit_str = SUIT_MAP_POKERLIB.get(suit, "?")
                    formatted_cards.append(f"{rank_str}{suit_str}")
                cards_str = f"( {' '.join(formatted_cards)} )"
            
            # Player Status Flags
            status = []
            if hasattr(p, 'is_folded') and p.is_folded: status.append("FOLDED")
            if hasattr(p, 'is_all_in') and p.is_all_in: status.append("ALL-IN")
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
            lines.append(f"{prefix} {name_str} ${money_val} {cards_str} {bet_str} {status_str}")
        
        # Action Context for the Acting Player
        lines.append("\n**Action Context:**")
        current_round_bet = 0  # Bet level for the current street
        if table_state.round and hasattr(table_state.round, 'turn_stake'):
            current_round_bet = table_state.round.turn_stake
        lines.append(f"- Current Bet to Match: ${current_round_bet}")
        
        amount_to_call = table_state.to_call  # Amount needed for THIS player to call
        lines.append(f"- Amount You Need to Call: ${amount_to_call}")
        acting_player_money = 'N/A'
        if acting_player_obj and hasattr(acting_player_obj, 'money'):
            acting_player_money = acting_player_obj.money
        lines.append(f"- Your Stack: ${acting_player_money}")
        
        # Possible Actions
        possible_actions = ["FOLD"]
        if table_state.can_check:
            possible_actions.append("CHECK")
        elif amount_to_call > 0 and (acting_player_obj and acting_player_money > 0):
            call_amount_str = min(amount_to_call, acting_player_money)  # Effective call amount
            possible_actions.append(f"CALL({call_amount_str})")
        if table_state.can_raise:
            possible_actions.append("RAISE")
        lines.append(f"- Possible Actions: {', '.join(possible_actions)}")
        
        # Task Prompt
        lines.append(f"\n**Task:** Respond ONLY with the action name: {', '.join(['CALL' if 'CALL(' in a else a for a in possible_actions])}.")
        lines.append("If RAISING, add ' AMOUNT: X' on the same line, where X is the integer amount to raise BY (the additional amount on top of any call). Example: 'RAISE AMOUNT: 20'")
        
        return "\n".join(lines)
    
    except Exception as e:
        print(f"Error during format_state_for_ai: {e}")
        return f"Error: Could not format game state - {e}"

def get_ai_action(table_state, player_id):
    """Gets an action from an AI player using the Together AI API."""
    player_name = table_state._get_player_name(player_id)
    model_name = table_state.ai_model_assignments.get(player_id)
    
    if not model_name:
        print(f"Error: No AI model assigned to player {player_name} (ID: {player_id}). Defaulting to FOLD.")
        return RoundPublicInId.FOLD, {}
    
    # Display thinking message
    model_short_name = DEFAULT_AI_MODEL_SHORT_NAMES.get(model_name, model_name.split('/')[-1])
    print(f"AI Thinking: {player_name} using {model_short_name}")
    
    # Format the game state for AI
    prompt = format_state_for_ai(table_state, player_id)
    if "Error:" in prompt:
        print(f"Error formatting game state for AI {player_name}: {prompt}. Defaulting to FOLD.")
        return RoundPublicInId.FOLD, {}
    
    # Set up messages for the AI
    round_id = table_state.round.id if table_state.round else 0
    history = table_state.ai_message_history[round_id][player_id]
    system_prompt = {"role": "system", "content": "You are a Texas Hold'em poker AI. Respond ONLY with the action name. If raising, add ' AMOUNT: X'."}
    messages = [system_prompt] + history + [{"role": "user", "content": prompt}]
    
    # Call the AI API
    ai_response_text = query_together_ai(
        model_name, 
        messages, 
        DEFAULT_SETTINGS['AI_TEMPERATURE'],
        DEFAULT_SETTINGS['AI_REQUEST_TIMEOUT'],
        DEFAULT_SETTINGS['AI_RETRY_DELAY']
    )
    
    if ai_response_text:
        print(f"AI Raw Response ({player_name}): {ai_response_text}")
        
        # Parse the AI's response into an action
        action_enum, action_kwargs = parse_ai_action(ai_response_text)
        
        # Validate the action
        player = table_state.seats.getPlayerById(player_id)
        if not player:
            print(f"Error: Could not find player object for ID {player_id}. Defaulting to FOLD.")
            return RoundPublicInId.FOLD, {}
        
        is_possible = False
        fallback_needed = False
        fallback_action = RoundPublicInId.FOLD
        fallback_kwargs = {}
        warn_msg = ""
        
        if action_enum == RoundPublicInId.FOLD: 
            is_possible = True
        elif action_enum == RoundPublicInId.CHECK:
            is_possible = table_state.can_check
            if not is_possible: 
                warn_msg = f"Warning: AI {player_name} chose CHECK when not possible (To Call: ${table_state.to_call})."
        elif action_enum == RoundPublicInId.CALL:
            is_possible = not table_state.can_check and table_state.to_call > 0 and player.money > 0
            if not is_possible: 
                warn_msg = f"Warning: AI {player_name} chose CALL when not possible or not necessary."
        elif action_enum == RoundPublicInId.RAISE:
            is_possible = table_state.can_raise
            if is_possible:
                raise_by = action_kwargs.get('raise_by', 0)
                max_raise = player.money - table_state.to_call
                min_r = table_state.min_raise
                if max_raise < min_r and player.money > table_state.to_call: 
                    min_r = max_raise
                is_all_in_raise = (table_state.to_call + raise_by) >= player.money
                
                if raise_by <= 0:
                    fallback_needed = True
                    warn_msg = f"Warning: AI {player_name} chose RAISE with invalid amount {raise_by} (<=0)."
                elif raise_by > max_raise:
                    fallback_needed = True
                    warn_msg = f"Warning: AI {player_name} chose RAISE {raise_by}, exceeding max possible ({max_raise})."
                    action_kwargs['raise_by'] = max_raise
                elif raise_by < min_r and not is_all_in_raise:
                    fallback_needed = True
                    warn_msg = f"Warning: AI {player_name} chose RAISE {raise_by}, below minimum ({min_r}) and not all-in."
                else: 
                    action_kwargs['raise_by'] = min(raise_by, max_raise)
            else:
                fallback_needed = True
                is_possible = False
                warn_msg = f"Warning: AI {player_name} chose RAISE when not allowed."
        
        if not is_possible or fallback_needed:
            if warn_msg:
                print(warn_msg)
            
            if not table_state.can_check and table_state.to_call > 0 and player.money > 0:
                fallback_action = RoundPublicInId.CALL
                fallback_kwargs = {}
                print("Fallback Action: CALL")
            elif table_state.can_check:
                fallback_action = RoundPublicInId.CHECK
                fallback_kwargs = {}
                print("Fallback Action: CHECK")
            else:
                fallback_action = RoundPublicInId.FOLD
                fallback_kwargs = {}
                print("Fallback Action: FOLD")
            
            action_enum = fallback_action
            action_kwargs = fallback_kwargs
        
        # Store the AI interaction in history
        history.append({"role": "user", "content": prompt})
        assistant_response = f"{action_enum.name}"
        if action_enum == RoundPublicInId.RAISE:
            assistant_response += f" AMOUNT: {action_kwargs['raise_by']}"
        history.append({"role": "assistant", "content": assistant_response})
        
        print(f"AI Validated Action ({player_name}): {action_enum.name} {action_kwargs}")
        return action_enum, action_kwargs
    else:
        print(f"Error: AI ({player_name}) failed to provide a response. Defaulting to FOLD.")
        return RoundPublicInId.FOLD, {}

class PokerGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        
        # Load settings or create default
        self.settings = self.load_settings()
        self.active_models = self.load_active_models()
        
        # Set up display
        display_flags = 0

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), display_flags)
        pygame.display.set_caption("NLPoker - Retro Edition")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = GameState.MAIN_MENU
        
        # Load assets
        self.load_assets()
        
        # Game variables
        self.table = None
        self.players = []
        self.round_count = 0
        self.menu_selection = 0
        self.message_log = []
        self.action_buttons = []
        self.slider_active = False
        self.slider_value = 0
        self.raise_amount = 0
        
        # Animation variables
        self.animations = []
        self.show_help = False
        
        # Set up the logger
        self.setup_logger()
    
    def load_settings(self):
        """Load settings from file or return defaults if not found."""
        try:
            with open('poker_settings.json', 'r') as f:
                settings = json.load(f)
                # Ensure all default settings exist in the loaded settings
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except (FileNotFoundError, json.JSONDecodeError):
            return DEFAULT_SETTINGS.copy()
    
    def save_settings(self):
        """Save settings to file."""
        with open('poker_settings.json', 'w') as f:
            json.dump(self.settings, f, indent=4)
    
    def load_active_models(self):
        """Load selected AI models from file or return defaults."""
        try:
            with open('active_models.json', 'r') as f:
                models = json.load(f)
                return models
        except (FileNotFoundError, json.JSONDecodeError):
            return DEFAULT_AI_MODELS.copy()
    
    def save_active_models(self):
        """Save selected AI models to file."""
        with open('active_models.json', 'w') as f:
            json.dump(self.active_models, f, indent=4)
    
    def load_assets(self):
        """Load fonts, images, and sounds."""
        # Load retro font
        self.font_path = os.path.join('assets', 'retro.ttf')
        pygame.freetype.init()
        try:
            self.title_font = pygame.freetype.Font(self.font_path, 40)  # Smaller title
            self.menu_font = pygame.freetype.Font(self.font_path, 32)   # Smaller menu text
            self.game_font = pygame.freetype.Font(self.font_path, 20)   # Smaller game text
            self.small_font = pygame.freetype.Font(self.font_path, 16)  # Smaller info text
            self.card_font = pygame.freetype.Font(self.font_path, 22)   # Font specifically for cards
        except:
            print("Error loading fonts. Using system font instead.")
            self.title_font = pygame.freetype.SysFont('monospace', 40)
            self.menu_font = pygame.freetype.SysFont('monospace', 32)
            self.game_font = pygame.freetype.SysFont('monospace', 20)
            self.small_font = pygame.freetype.SysFont('monospace', 16)
            self.card_font = pygame.freetype.SysFont('monospace', 22)
        
        # Set up simple card graphics and chip graphics
        self.card_width = 80
        self.card_height = 120
        self.chip_radius = 25
        
        # Load sound effects
        try:
            self.sounds = {
                'click': pygame.mixer.Sound(os.path.join('assets', 'click.wav')),
                'deal': pygame.mixer.Sound(os.path.join('assets', 'click.wav')),  # Reusing for now
                'chips': pygame.mixer.Sound(os.path.join('assets', 'click.wav')), # Reusing for now
                'win': pygame.mixer.Sound(os.path.join('assets', 'click.wav')),   # Reusing for now
            }
        except:
            print("Error loading sound effects. Sound will be disabled.")
            self.settings['SOUND_EFFECTS'] = False
    
    def setup_logger(self):
        """Set up logging for the game."""
        self.log_messages = []
    
    def log(self, message, level="INFO"):
        """Add a message to the log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] [{level}] {message}")
        # Keep only the last 100 messages
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)
    
    def play_sound(self, sound_name):
        """Play a sound effect if enabled."""
        if self.settings['SOUND_EFFECTS'] and sound_name in self.sounds:
            self.sounds[sound_name].play()
    
    def init_table(self):
        """Initialize the poker table and players."""
        # Calculate how many players we need
        num_ai_models = len(self.active_models)
        num_players = num_ai_models
        if not self.settings['AI_ONLY_MODE']:
            num_players += 1
        
        # Create table
        self.table = PyGameTable(
            _id=0,
            seats=PlayerSeats([None] * num_players),
            buyin=self.settings['BUYIN'],
            small_blind=self.settings['SMALL_BLIND'],
            big_blind=self.settings['BIG_BLIND']
        )
        
        # Create players
        self.players = []
        player_id_counter = 1
        
        # Add human player if not AI-only mode
        if not self.settings['AI_ONLY_MODE']:
            human_player = Player(
                table_id=self.table.id,
                _id=self.settings['HUMAN_PLAYER_ID'],
                name=self.settings['HUMAN_PLAYER_NAME'],
                money=self.settings['BUYIN']
            )
            self.players.append(human_player)
            player_id_counter = max(player_id_counter, self.settings['HUMAN_PLAYER_ID'] + 1)
        
        # Add AI players
        for i, model in enumerate(self.active_models):
            ai_short_name = DEFAULT_AI_MODEL_SHORT_NAMES.get(model, f"AI_{player_id_counter}")
            ai_player = Player(
                table_id=self.table.id,
                _id=player_id_counter,
                name=ai_short_name,
                money=self.settings['BUYIN']
            )
            self.players.append(ai_player)
            player_id_counter += 1
        
        # Shuffle players for seating
        random.shuffle(self.players)
        
        # Seat players
        for p in self.players:
            self.table.publicIn(p.id, TablePublicInId.BUYIN, player=p)
        
        # Assign AI models
        self.table.assign_ai_models()
        
        # Reset game variables
        self.round_count = 0
        self.message_log = []
    
    def start_round(self):
        """Start a new poker round."""
        if not self.table:
            self.init_table()
        
        active_players = self.table.seats.getPlayerGroup()
        if len(active_players) < 2:
            self.log("Not enough players to start a round", "ERROR")
            return
        
        self.round_count += 1
        initiator = active_players[0]
        
        # Start round
        self.table.publicIn(initiator.id, TablePublicInId.STARTROUND, round_id=self.round_count)
        
        if not self.table.round:
            self.log(f"Round {self.round_count} failed to initialize", "ERROR")
            return
        
        self.log(f"Started Round {self.round_count}")
        self.state = GameState.PLAYING
    
    def draw_menu(self):
        """Draw the main menu screen."""
        # Background
        self.screen.fill(POKER_DARK_GREEN)
        
        # Title
        title_text = "NLPOKER - RETRO EDITION"
        title_surf, title_rect = self.title_font.render(title_text, WHITE)
        title_rect.center = (self.screen_width // 2, 120)
        self.screen.blit(title_surf, title_rect)
        
        # Menu options
        menu_options = [
            "START GAME",
            "SETTINGS",
            "MODEL SELECT",
            "QUIT"
        ]
        
        for i, option in enumerate(menu_options):
            if i == self.menu_selection:
                color = GOLD
                text = f"> {option} <"
            else:
                color = WHITE
                text = option
            
            option_surf, option_rect = self.menu_font.render(text, color)
            option_rect.center = (self.screen_width // 2, 250 + i * 60)
            self.screen.blit(option_surf, option_rect)
        
        # Game mode indicator
        mode_text = "GAME MODE: AI ONLY" if self.settings['AI_ONLY_MODE'] else f"GAME MODE: HUMAN ({self.settings['HUMAN_PLAYER_NAME']}) VS AI"
        mode_surf, mode_rect = self.small_font.render(mode_text, CYAN)
        mode_rect.center = (self.screen_width // 2, 550)
        self.screen.blit(mode_surf, mode_rect)
        
        # AI models selected
        models_text = f"AI MODELS: {len(self.active_models)} SELECTED"
        models_surf, models_rect = self.small_font.render(models_text, CYAN)
        models_rect.center = (self.screen_width // 2, 580)
        self.screen.blit(models_surf, models_rect)
        
        
        # Version info
        version_text = "V1.0 - 8-BIT POKER"
        version_surf, version_rect = self.small_font.render(version_text, LIGHT_GRAY)
        version_rect.bottomright = (self.screen_width - 20, self.screen_height - 20)
        self.screen.blit(version_surf, version_rect)
    
    def draw_settings(self):
        """Draw the settings screen."""
        # Background
        self.screen.fill(POKER_DARK_GREEN)
        
        # Title
        title_text = "SETTINGS"
        title_surf, title_rect = self.title_font.render(title_text, WHITE)
        title_rect.center = (SCREEN_WIDTH // 2, 70)  # Moved up from 80
        self.screen.blit(title_surf, title_rect)
        
        # Settings options with new AI delay option
        settings_options = [
            f"BUYIN: ${self.settings['BUYIN']}",
            f"SMALL BLIND: ${self.settings['SMALL_BLIND']}",
            f"BIG BLIND: ${self.settings['BIG_BLIND']}",
            f"AI ONLY MODE: {'ON' if self.settings['AI_ONLY_MODE'] else 'OFF'}",
            f"HUMAN NAME: {self.settings['HUMAN_PLAYER_NAME']}",
            f"SHOW PROBABILITIES: {'ON' if self.settings['SHOW_PROBABILITIES'] else 'OFF'}",
            f"SIMULATIONS: {self.settings['PROBABILITY_SIMULATIONS']}",
            f"AI TEMPERATURE: {self.settings['AI_TEMPERATURE']}",
            f"SOUND EFFECTS: {'ON' if self.settings['SOUND_EFFECTS'] else 'OFF'}",
            f"AI THINK DELAY: {self.settings['AI_THINK_DELAY']}s",  # New setting for AI delay
            f"ANIMATION SPEED: {self.settings['ANIMATION_SPEED']}",
            "SAVE AND RETURN"
        ]
        
        # Adjust vertical spacing to fit all options
        vertical_spacing = 42  # Reduced from 50
        
        for i, option in enumerate(settings_options):
            if i == self.menu_selection:
                color = GOLD
                text = f"> {option} <"
            else:
                color = WHITE
                text = option
            
            option_surf, option_rect = self.menu_font.render(text, color)
            option_rect.center = (SCREEN_WIDTH // 2, 130 + i * vertical_spacing)
            self.screen.blit(option_surf, option_rect)
        
        # Instructions
        instruction_text = "ARROW KEYS TO NAVIGATE, ENTER TO SELECT, SPACE/LEFT/RIGHT TO CHANGE VALUES"
        instruction_surf, instruction_rect = self.small_font.render(instruction_text, LIGHT_GRAY)
        instruction_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40)
        self.screen.blit(instruction_surf, instruction_rect)
    
    def draw_model_select(self):
        """Draw the AI model selection screen."""
        # Background
        self.screen.fill(POKER_DARK_GREEN)
        
        # Title
        title_text = "AI MODEL SELECT"
        title_surf, title_rect = self.title_font.render(title_text, WHITE)
        title_rect.center = (SCREEN_WIDTH // 2, 80)
        self.screen.blit(title_surf, title_rect)
        
        # Get all available models
        all_models = list(DEFAULT_AI_MODEL_SHORT_NAMES.keys())
        
        # Draw instructions
        instruction_text = "SPACE TO TOGGLE SELECTION - ENTER TO SAVE AND RETURN"
        instruction_surf, instruction_rect = self.small_font.render(instruction_text, LIGHT_GRAY)
        instruction_rect.center = (SCREEN_WIDTH // 2, 130)
        self.screen.blit(instruction_surf, instruction_rect)
        
        # Current selection count
        count_text = f"SELECTED: {len(self.active_models)} MODELS"
        count_surf, count_rect = self.menu_font.render(count_text, CYAN)
        count_rect.center = (SCREEN_WIDTH // 2, 170)
        self.screen.blit(count_surf, count_rect)
        
        # Draw models list
        start_y = 220
        items_per_page = 10
        page = self.menu_selection // items_per_page
        start_idx = page * items_per_page
        
        for i in range(start_idx, min(start_idx + items_per_page, len(all_models))):
            model = all_models[i]
            short_name = DEFAULT_AI_MODEL_SHORT_NAMES[model]
            is_selected = model in self.active_models
            
            if i == self.menu_selection:
                color = GOLD
                prefix = "> "
                suffix = " <"
            else:
                color = WHITE
                prefix = "  "
                suffix = "  "
            
            status = "[X]" if is_selected else "[ ]"
            model_text = f"{prefix}{status} {short_name}{suffix}"
            
            model_surf, model_rect = self.menu_font.render(model_text, color)
            model_rect.topleft = (SCREEN_WIDTH // 4, start_y + (i - start_idx) * 45)
            self.screen.blit(model_surf, model_rect)
        
        # Page indicator
        total_pages = (len(all_models) + items_per_page - 1) // items_per_page
        page_text = f"PAGE {page + 1}/{total_pages}"
        page_surf, page_rect = self.small_font.render(page_text, LIGHT_GRAY)
        page_rect.bottomright = (SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20)
        self.screen.blit(page_surf, page_rect)
    
    def draw_card(self, card, x, y, face_up=True, scale=1.0):
        """Draw a card at the specified position."""
        # Scale dimensions
        width = int(self.card_width * scale)
        height = int(self.card_height * scale)
        
        # Card background
        if face_up and card:
            # White background with black border
            pygame.draw.rect(self.screen, WHITE, (x, y, width, height), border_radius=3)
            pygame.draw.rect(self.screen, BLACK, (x, y, width, height), width=2, border_radius=3)
            
            # Draw rank and suit
            rank, suit = card
            rank_str = RANK_MAP_POKERLIB.get(rank, "?")
            suit_str = SUIT_MAP_POKERLIB.get(suit, "?")
            color = SUIT_COLOR_MAP.get(suit, WHITE)
            
            # Create card text (like "4S" for 4 of Spades)
            card_text = f"{rank_str}{suit_str}"
            
            # Use the dedicated card font to ensure it fits
            card_surf, card_rect = self.card_font.render(card_text, color)
            
            # Make sure text fits inside card
            if card_rect.width > width - 10:  # Ensure margin around text
                # If too large, use a smaller font
                card_surf, card_rect = self.small_font.render(card_text, color)
            
            # Center the text
            card_rect.center = (x + width // 2, y + height // 2)
            self.screen.blit(card_surf, card_rect)
        else:
            # Card back - simplified design
            pygame.draw.rect(self.screen, DARK_GRAY, (x, y, width, height), border_radius=3)
            pygame.draw.rect(self.screen, BLACK, (x, y, width, height), width=2, border_radius=3)
            
            # Simple pattern for card back
            margin = 5
            pattern_rect = pygame.Rect(x + margin, y + margin, width - 2*margin, height - 2*margin)
            pygame.draw.rect(self.screen, MAGENTA, pattern_rect, border_radius=2)
            
            # Cross pattern on card back
            pygame.draw.line(self.screen, CYAN, (x + margin, y + margin), 
                            (x + width - margin, y + height - margin), 2)
            pygame.draw.line(self.screen, CYAN, (x + margin, y + height - margin), 
                            (x + width - margin, y + margin), 2)
    
    def draw_chip(self, x, y, amount, color=GOLD):
        """Draw a poker chip with an amount."""
        # Draw chip
        pygame.draw.circle(self.screen, color, (x, y), self.chip_radius)
        pygame.draw.circle(self.screen, BLACK, (x, y), self.chip_radius, 2)
        
        # Draw amount
        amount_text = f"${amount}"
        amount_surf, amount_rect = self.small_font.render(amount_text, BLACK)
        amount_rect.center = (x, y)
        self.screen.blit(amount_surf, amount_rect)
    
    def draw_player(self, player, pos_x, pos_y, is_active=False, is_human=False, box_width=170, box_height=100):
        """Draw a player with their info and cards."""
        # Draw player box
        box_color = GOLD if is_active else DARK_GRAY
        inner_color = LIGHT_GRAY if is_human else DARK_GRAY
        
        pygame.draw.rect(self.screen, box_color, (pos_x, pos_y, box_width, box_height), border_radius=6)
        pygame.draw.rect(self.screen, inner_color, (pos_x + 4, pos_y + 4, box_width - 8, box_height - 8), border_radius=4)
        
        # Draw player name
        name_text = player.name if hasattr(player, 'name') else f"P{player.id}"
        # Truncate name if too long
        if len(name_text) > 12:
            name_text = name_text[:10] + ".."
        name_surf, name_rect = self.small_font.render(name_text, WHITE)
        name_rect.topleft = (pos_x + 8, pos_y + 8)
        self.screen.blit(name_surf, name_rect)
        
        # Draw player money
        money_val = player.money if hasattr(player, 'money') else 0
        money_text = f"${money_val}"
        money_surf, money_rect = self.small_font.render(money_text, GREEN)
        money_rect.topleft = (pos_x + 8, pos_y + 27) # Adjusted y position
        self.screen.blit(money_surf, money_rect)
        
        # Check if player is folded or all-in
        is_folded = hasattr(player, 'is_folded') and player.is_folded
        is_all_in = hasattr(player, 'is_all_in') and player.is_all_in
        
        # Draw player status flags
        status = []
        if is_folded: status.append("FOLDED")
        if is_all_in: status.append("ALL-IN")
        
        if status:
            status_text = " ".join(status)
            status_surf, status_rect = self.small_font.render(status_text, RED)
            status_rect.topleft = (pos_x + 8, pos_y + 46) # Adjusted y position
            self.screen.blit(status_surf, status_rect)
        
        # Card dimensions - smaller scale
        card_scale = 0.4  # Reduced from 0.45
        card_width = int(self.card_width * card_scale)
        card_height = int(self.card_height * card_scale)
        card_spacing = 4   # Reduced from 5
        
        # Calculate positions to place cards side by side
        card1_x = pos_x + box_width - (card_width * 2) - card_spacing - 8
        card1_y = pos_y + 45 # Adjusted y position
        card2_x = card1_x + card_width + card_spacing
        card2_y = card1_y
        
        # Draw cards based on player state
        player_id = player.id if hasattr(player, 'id') else 0
        
        if player_id in self.table._player_cards:
            cards = self.table._player_cards[player_id]
            
            if is_folded:
                # For folded players, show a distinctive "folded" display
                # Draw red X over cards or gray them out
                folded_bg = pygame.Rect(card1_x, card1_y, (card_width * 2) + card_spacing, card_height)
                pygame.draw.rect(self.screen, (80, 80, 80), folded_bg, border_radius=3)
                pygame.draw.rect(self.screen, (130, 0, 0), folded_bg, width=2, border_radius=3)
                
                # Draw "X" over cards
                pygame.draw.line(self.screen, (200, 0, 0), 
                                (folded_bg.left + 5, folded_bg.top + 5), 
                                (folded_bg.right - 5, folded_bg.bottom - 5), 2)
                pygame.draw.line(self.screen, (200, 0, 0), 
                                (folded_bg.right - 5, folded_bg.top + 5), 
                                (folded_bg.left + 5, folded_bg.bottom - 5), 2)
                
                # Draw "FOLD" text with smaller font
                folded_surf, folded_rect = self.small_font.render("FOLD", RED)
                folded_rect.center = folded_bg.center
                self.screen.blit(folded_surf, folded_rect)
            else:
                # Determine if cards should be face up
                show_cards = False
                if not self.settings['AI_ONLY_MODE'] and player_id == self.settings['HUMAN_PLAYER_ID']:
                    show_cards = True
                elif self.settings['AI_ONLY_MODE']:
                    show_cards = True
                elif hasattr(self.table, 'round_over_flag') and self.table.round_over_flag and hasattr(self.table, 'showdown_occurred') and self.table.showdown_occurred:
                    show_cards = True
                
                # Draw both cards
                if len(cards) >= 2:
                    self.draw_card(cards[0], card1_x, card1_y, face_up=show_cards, scale=card_scale)
                    self.draw_card(cards[1], card2_x, card2_y, face_up=show_cards, scale=card_scale)
        else:
            # No cards yet - draw placeholders
            for i in range(2):
                pygame.draw.rect(self.screen, (70, 70, 70), 
                                (card1_x + i * (card_width + card_spacing), card1_y, 
                                card_width, card_height), 
                                width=1, border_radius=3)
        
        # Draw probability if enabled (only for active players)
        if self.settings['SHOW_PROBABILITIES'] and not is_folded:
            if player_id in self.table._player_probabilities:
                prob = self.table._player_probabilities[player_id]
                prob_text = f"WIN:{prob:.1f}%"
                prob_surf, prob_rect = self.small_font.render(prob_text, CYAN)
                prob_rect.topleft = (pos_x + 8, pos_y + box_height - 22) # Position at bottom
                self.screen.blit(prob_surf, prob_rect)
    
    def draw_board(self):
        """Draw the poker table and board cards."""
        # Draw table background - adjusted positioning
        table_radius = 210  # Smaller table (from 225)
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 + 10  # Less shifting down (from 30)
        
        # Draw poker table
        pygame.draw.circle(self.screen, BLACK, (center_x, center_y), table_radius + 10)
        pygame.draw.circle(self.screen, POKER_GREEN, (center_x, center_y), table_radius)
        pygame.draw.circle(self.screen, POKER_DARK_GREEN, (center_x, center_y), table_radius - 20)
        
        # Draw pot
        if hasattr(self.table, 'round') and hasattr(self.table.round, 'pot_size'):
            pot_total = sum(self.table.round.pot_size)
            if pot_total > 0:
                # Draw chips for the pot
                self.draw_chip(center_x, center_y - 30, pot_total)
                
                # Draw pot text
                pot_text = f"POT: ${pot_total}"
                pot_surf, pot_rect = self.game_font.render(pot_text, WHITE)
                pot_rect.center = (center_x, center_y - 65) # Moved up slightly
                self.screen.blit(pot_surf, pot_rect)
        
        # Draw board cards
        board_cards = []
        if hasattr(self.table, 'round') and hasattr(self.table.round, 'board'):
            board_cards = [tuple(c) for c in self.table.round.board if isinstance(c, (list,tuple)) and len(c)==2]
        
        # Calculate position for cards - use smaller cards with less spacing
        card_scale = 0.7  # Reduced from 0.8
        scaled_card_width = int(self.card_width * card_scale)
        card_spacing = 6  # Reduced from 8
        
        num_cards = len(board_cards)
        if num_cards > 0:
            total_width = (num_cards * scaled_card_width) + ((num_cards - 1) * card_spacing)
            start_x = center_x - (total_width // 2)
            
            for i, card in enumerate(board_cards):
                self.draw_card(card, start_x + (i * (scaled_card_width + card_spacing)), 
                               center_y + 5, scale=card_scale)  # Moved cards up slightly (from 10)
        
        # Draw round info with smaller text
        if hasattr(self.table, 'round') and hasattr(self.table.round, 'id') and hasattr(self.table.round, 'turn'):
            round_id = self.table.round.id
            turn_name = self.table.round.turn.name
            
            round_text = f"ROUND: {round_id} - {turn_name}"
            round_surf, round_rect = self.small_font.render(round_text, WHITE)
            round_rect.center = (center_x, center_y + 120)  # Reduced from 140
            self.screen.blit(round_surf, round_rect)
    
    def draw_players_around_table(self):
        """Draw all players positioned around the table."""
        if not self.table or not hasattr(self.table, 'round'):
            return
        
        players = self.table.round.players if hasattr(self.table.round, 'players') else []
        if not players:
            return
        
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 + 10  # Shifted up slightly to center better
        radius = 300  # Slightly smaller radius for positioning
        
        # Get the currently active player
        current_player_id = self.table._current_action_player_id if hasattr(self.table, '_current_action_player_id') else None
        
        # Use smaller player box dimensions 
        box_width = 170  # Reduced from 190
        box_height = 100  # Reduced from 110
        
        # Dynamically adjust positions based on number of players to avoid overlap
        num_players = len([p for p in players if p and hasattr(p, 'id')])
        
        # Define offsets based on player count to avoid overlap
        if num_players <= 2:
            position_offsets = [
                (0, -radius * 0.9),             # Top
                (0, radius * 0.9),              # Bottom
            ]
        elif num_players <= 4:
            position_offsets = [
                (0, -radius * 0.9),             # Top
                (radius * 0.9, 0),              # Right
                (0, radius * 0.9),              # Bottom
                (-radius * 0.9, 0),             # Left
            ]
        elif num_players <= 6:
            position_offsets = [
                (0, -radius),                    # Top
                (radius * 0.85, -radius * 0.45), # Top-Right
                (radius * 0.85, radius * 0.45),  # Bottom-Right
                (0, radius),                     # Bottom
                (-radius * 0.85, radius * 0.45), # Bottom-Left
                (-radius * 0.85, -radius * 0.45),# Top-Left
            ]
        else:
            # Calculate positions evenly around the circle with slight adjustments
            # to prevent boxes from going off-screen
            position_offsets = []
            for i in range(num_players):
                angle = -math.pi/2 + (i * 2 * math.pi / num_players)
                # Add some padding to keep players from the extreme edges
                offset_x = radius * 0.9 * math.cos(angle) 
                offset_y = radius * 0.9 * math.sin(angle)
                
                # Adjust vertical positions slightly to improve distribution
                if abs(offset_y) > radius * 0.85:
                    offset_y *= 0.9  # Bring in extreme top/bottom positions
                    
                position_offsets.append((offset_x, offset_y))
        
        player_count = 0
        for idx, player in enumerate(players):
            if not player or not hasattr(player, 'id'):
                continue
            
            if player_count >= len(position_offsets):
                continue  # Skip if we don't have a position for this player
            
            # Get offset from the predefined positions
            offset_x, offset_y = position_offsets[player_count]
            pos_x = int(center_x + offset_x)
            pos_y = int(center_y + offset_y)
            player_count += 1
            
            # Draw the player with their current state
            player_id = player.id
            is_active = (player_id == current_player_id)
            is_human = (not self.settings['AI_ONLY_MODE'] and player_id == self.settings['HUMAN_PLAYER_ID'])
            
            # Adjust position so player is centered on their spot
            pos_x -= box_width // 2
            pos_y -= box_height // 2
            
            self.draw_player(player, pos_x, pos_y, is_active, is_human, box_width, box_height)
            
            # Draw player's current bet if any
            if hasattr(player, 'turn_stake') and hasattr(self.table.round, 'turn'):
                current_turn_enum = self.table.round.turn
                if (isinstance(player.turn_stake, list) and
                    hasattr(current_turn_enum, 'value') and
                    len(player.turn_stake) > current_turn_enum.value):
                    
                    turn_stake_val = player.turn_stake[current_turn_enum.value]
                    if turn_stake_val > 0:
                        # Calculate position for the chips between player and center
                        # Bring chips closer to the player for better visibility
                        chip_distance = 0.65  # 65% of the way from center to player
                        chip_x = int(center_x + offset_x * chip_distance)
                        chip_y = int(center_y + offset_y * chip_distance)
                        
                        self.draw_chip(chip_x, chip_y, turn_stake_val, GOLD)
    
    def draw_action_buttons(self):
        """Draw buttons for player actions."""
        if not self.table or not hasattr(self.table, '_current_action_player_id'):
            return
        
        # Only show buttons for human player's turn
        if (self.settings['AI_ONLY_MODE'] or 
            self.table._current_action_player_id != self.settings['HUMAN_PLAYER_ID']):
            return
        
        button_width = 120
        button_height = 50
        button_spacing = 20
        button_y = SCREEN_HEIGHT - button_height - 20
        
        # Determine available actions
        actions = ["FOLD"]
        self.action_buttons = []
        
        # Position for the first button - moved to the left side with margin
        start_x = 20  # Left margin of 20px
        current_x = start_x
        
        # Draw FOLD button
        fold_rect = pygame.Rect(current_x, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, RED, fold_rect, border_radius=5)
        pygame.draw.rect(self.screen, BLACK, fold_rect, width=2, border_radius=5)
        
        fold_text = "FOLD"
        fold_surf, fold_text_rect = self.game_font.render(fold_text, WHITE)
        fold_text_rect.center = fold_rect.center
        self.screen.blit(fold_surf, fold_text_rect)
        
        self.action_buttons.append(('FOLD', fold_rect))
        current_x += button_width + button_spacing
        
        # Draw CHECK/CALL button
        check_color = GREEN
        check_text = "CHECK" if self.table.can_check else "CALL"
        if not self.table.can_check:
            check_text = f"CALL ${self.table.to_call}"
        
        check_rect = pygame.Rect(current_x, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, check_color, check_rect, border_radius=5)
        pygame.draw.rect(self.screen, BLACK, check_rect, width=2, border_radius=5)
        
        check_surf, check_text_rect = self.game_font.render(check_text, WHITE)
        check_text_rect.center = check_rect.center
        self.screen.blit(check_surf, check_text_rect)
        
        self.action_buttons.append((check_text, check_rect))
        current_x += button_width + button_spacing
        
        # Draw RAISE button if available
        if self.table.can_raise:
            raise_rect = pygame.Rect(current_x, button_y, button_width, button_height)
            pygame.draw.rect(self.screen, BLUE, raise_rect, border_radius=5)
            pygame.draw.rect(self.screen, BLACK, raise_rect, width=2, border_radius=5)
            
            raise_text = f"RAISE"
            raise_surf, raise_text_rect = self.game_font.render(raise_text, WHITE)
            raise_text_rect.center = raise_rect.center
            self.screen.blit(raise_surf, raise_text_rect)
            
            self.action_buttons.append(('RAISE', raise_rect))
            
            # Draw raise slider if active
            if self.slider_active:
                self.draw_raise_slider()
    
    def draw_raise_slider(self):
        """Draw a slider for selecting raise amount."""
        slider_y = SCREEN_HEIGHT - 120
        slider_width = 400
        slider_height = 30
        slider_x = (SCREEN_WIDTH - slider_width) // 2
        
        # Get player object
        player = self.table.seats.getPlayerById(self.settings['HUMAN_PLAYER_ID'])
        if not player:
            return
        
        # Calculate raise limits
        min_raise = self.table.big_blind
        max_raise = player.money - self.table.to_call
        
        if max_raise < min_raise and player.money > self.table.to_call:
            min_raise = max_raise
        
        if max_raise <= 0:
            # Can't raise
            return
        
        # Draw slider background
        slider_bg = pygame.Rect(slider_x, slider_y, slider_width, slider_height)
        pygame.draw.rect(self.screen, DARK_GRAY, slider_bg, border_radius=5)
        pygame.draw.rect(self.screen, BLACK, slider_bg, width=2, border_radius=5)
        
        # Draw slider position
        slider_pos = int(slider_x + (self.slider_value / 100) * slider_width)
        pygame.draw.circle(self.screen, WHITE, (slider_pos, slider_y + slider_height // 2), 15)
        pygame.draw.circle(self.screen, BLACK, (slider_pos, slider_y + slider_height // 2), 15, 2)
        
        # Calculate actual raise amount
        self.raise_amount = int(min_raise + ((max_raise - min_raise) * (self.slider_value / 100)))
        
        # Draw raise amount
        amount_text = f"RAISE TO: ${self.table.to_call + self.raise_amount}"
        amount_surf, amount_rect = self.game_font.render(amount_text, WHITE)
        amount_rect.center = (SCREEN_WIDTH // 2, slider_y - 25)
        self.screen.blit(amount_surf, amount_rect)
        
        # Draw min/max labels
        min_text = f"MIN: ${min_raise}"
        min_surf, min_rect = self.small_font.render(min_text, LIGHT_GRAY)
        min_rect.bottomleft = (slider_x, slider_y - 5)
        self.screen.blit(min_surf, min_rect)
        
        max_text = f"MAX: ${max_raise}"
        max_surf, max_rect = self.small_font.render(max_text, LIGHT_GRAY)
        max_rect.bottomright = (slider_x + slider_width, slider_y - 5)
        self.screen.blit(max_surf, max_rect)
        
        # Draw confirm/cancel buttons
        confirm_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, slider_y + slider_height + 15, 100, 40)
        pygame.draw.rect(self.screen, GREEN, confirm_rect, border_radius=5)
        pygame.draw.rect(self.screen, BLACK, confirm_rect, width=2, border_radius=5)
        
        confirm_text = "CONFIRM"
        confirm_surf, confirm_rect_text = self.small_font.render(confirm_text, WHITE)
        confirm_rect_text.center = confirm_rect.center
        self.screen.blit(confirm_surf, confirm_rect_text)
        
        cancel_rect = pygame.Rect(SCREEN_WIDTH // 2 + 20, slider_y + slider_height + 15, 100, 40)
        pygame.draw.rect(self.screen, RED, cancel_rect, border_radius=5)
        pygame.draw.rect(self.screen, BLACK, cancel_rect, width=2, border_radius=5)
        
        cancel_text = "CANCEL"
        cancel_surf, cancel_rect_text = self.small_font.render(cancel_text, WHITE)
        cancel_rect_text.center = cancel_rect.center
        self.screen.blit(cancel_surf, cancel_rect_text)
        
        # Add buttons to action buttons list for click detection
        self.action_buttons.append(('CONFIRM_RAISE', confirm_rect))
        self.action_buttons.append(('CANCEL_RAISE', cancel_rect))
    
    def draw_message_log(self):
        """Draw the game message log."""
        if not self.message_log:
            return
        
        log_width = 300
        log_height = 200
        log_x = 20
        log_y = 20
        
        # Draw log background
        pygame.draw.rect(self.screen, (0, 0, 0, 128), (log_x, log_y, log_width, log_height), border_radius=5)
        pygame.draw.rect(self.screen, BLACK, (log_x, log_y, log_width, log_height), width=2, border_radius=5)
        
        # Draw log title
        title_surf, title_rect = self.small_font.render("GAME MESSAGES", WHITE)
        title_rect.topleft = (log_x + 10, log_y + 5)
        self.screen.blit(title_surf, title_rect)
        
        # Draw messages (most recent at the bottom)
        visible_messages = self.message_log[-8:]  # Show only the most recent 8 messages
        for i, msg in enumerate(visible_messages):
            msg_surf, msg_rect = self.small_font.render(msg, LIGHT_GRAY)
            msg_rect.topleft = (log_x + 10, log_y + 30 + (i * 20))
            self.screen.blit(msg_surf, msg_rect)
    
    def draw_help_overlay(self):
        """Draw a help overlay with game controls and information."""
        if not self.show_help:
            return
        
        # Semi-transparent background overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Help window
        help_width = 600
        help_height = 500
        help_x = (SCREEN_WIDTH - help_width) // 2
        help_y = (SCREEN_HEIGHT - help_height) // 2
        
        pygame.draw.rect(self.screen, DARK_GRAY, (help_x, help_y, help_width, help_height), border_radius=10)
        pygame.draw.rect(self.screen, BLACK, (help_x, help_y, help_width, help_height), width=3, border_radius=10)
        
        # Title
        title_surf, title_rect = self.title_font.render("HELP & CONTROLS", WHITE)
        title_rect.center = (SCREEN_WIDTH // 2, help_y + 40)
        self.screen.blit(title_surf, title_rect)
        
        # Help content
        help_content = [
            "GAME CONTROLS:",
            "",
            "ESC - Exit current screen / Show menu",
            "H - Toggle this help screen",
            "",
            "DURING GAME:",
            "MOUSE - Select action buttons",
            "SPACE - Continue to next round",
            "",
            "IN MENUS:",
            "ARROW KEYS - Navigate options",
            "ENTER - Select option",
            "SPACE/LEFT/RIGHT - Change values",
            "",
            "PRESS H TO CLOSE"
        ]
        
        for i, line in enumerate(help_content):
            line_surf, line_rect = self.game_font.render(line, WHITE)
            line_rect.topleft = (help_x + 50, help_y + 100 + (i * 30))
            self.screen.blit(line_surf, line_rect)
    
    def draw_game(self):
        """Draw the main game screen."""
        # Draw background
        self.screen.fill(POKER_DARK_GREEN)
        
        # Draw poker table and board
        self.draw_board()
        
        # Draw players around the table
        self.draw_players_around_table()
        
        # Draw action buttons if needed
        self.draw_action_buttons()
        
        # Draw message log
        self.draw_message_log()
        
        # Draw help button
        help_text = "PRESS H FOR HELP"
        help_surf, help_rect = self.small_font.render(help_text, WHITE)
        help_rect.bottomright = (SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20)
        self.screen.blit(help_surf, help_rect)
        
        # Draw help overlay if active
        self.draw_help_overlay()
    
    def draw_round_end(self):
        """Draw the round end screen with summary and continue option."""
        # First draw the game state
        self.draw_game()
        
        # Then overlay with round summary
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        # Round end panel
        panel_width = 460  # Slightly smaller panel
        panel_height = 380
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        pygame.draw.rect(self.screen, BLACK, (panel_x, panel_y, panel_width, panel_height), width=3, border_radius=10)
        
        # Title - using smaller font
        title_text = f"ROUND {self.round_count} COMPLETE"
        title_surf, title_rect = self.menu_font.render(title_text, GOLD)  # Menu font instead of title
        title_rect.center = (self.screen_width // 2, panel_y + 40)
        self.screen.blit(title_surf, title_rect)
        
        # List winners - smaller text
        winners_text = "WINNER(S):"
        winners_surf, winners_rect = self.game_font.render(winners_text, WHITE)  # Game font instead of menu
        winners_rect.topleft = (panel_x + 40, panel_y + 90)
        self.screen.blit(winners_surf, winners_rect)
        
        # Show winner messages in a more organized layout
        if hasattr(self.table, 'pending_winner_messages'):
            y_offset = 130
            for i, msg in enumerate(self.table.pending_winner_messages):
                if "wins $" in msg:
                    # Extract and format winner info
                    parts = msg.split("wins $")
                    if len(parts) == 2:
                        # Get name, amount, and hand
                        winner_name = parts[0].strip().replace("[WINNER]", "").strip()
                        amount_parts = parts[1].split(" ", 1)
                        amount = amount_parts[0].strip()
                        hand = amount_parts[1] if len(amount_parts) > 1 else ""
                        
                        # Line 1: Winner name
                        name_surf, name_rect = self.game_font.render(winner_name, CYAN)
                        name_rect.topleft = (panel_x + 60, panel_y + y_offset)
                        self.screen.blit(name_surf, name_rect)
                        
                        # Line 2: Amount won
                        amount_text = f"Won: ${amount}"
                        amount_surf, amount_rect = self.game_font.render(amount_text, GREEN)
                        amount_rect.topleft = (panel_x + 60, panel_y + y_offset + 25)
                        self.screen.blit(amount_surf, amount_rect)
                        
                        # Line 3: Hand
                        if hand:
                            hand_surf, hand_rect = self.small_font.render(hand, LIGHT_GRAY)
                            hand_rect.topleft = (panel_x + 60, panel_y + y_offset + 50)
                            self.screen.blit(hand_surf, hand_rect)
                        
                        # Space for next winner (if any)
                        y_offset += 80  # Increased spacing between winners
        
        # Continue buttons - smaller text
        continue_text = "PRESS SPACE TO CONTINUE"
        continue_surf, continue_rect = self.game_font.render(continue_text, WHITE)
        continue_rect.center = (self.screen_width // 2, panel_y + panel_height - 70)
        self.screen.blit(continue_surf, continue_rect)
        
        # Exit button
        exit_text = "PRESS ENTER TO END GAME"
        exit_surf, exit_rect = self.game_font.render(exit_text, YELLOW)
        exit_rect.center = (self.screen_width // 2, panel_y + panel_height - 40)
        self.screen.blit(exit_surf, exit_rect)
    
    def process_menu_input(self, event):
        """Process input for the main menu."""
        if event.type == KEYDOWN:
            if event.key == K_UP:
                self.menu_selection = (self.menu_selection - 1) % 4
                self.play_sound('click')
            elif event.key == K_DOWN:
                self.menu_selection = (self.menu_selection + 1) % 4
                self.play_sound('click')
            elif event.key == K_RETURN:
                self.play_sound('click')
                if self.menu_selection == 0:  # Start Game
                    self.init_table()
                    self.start_round()
                elif self.menu_selection == 1:  # Settings
                    self.state = GameState.SETTINGS
                    self.menu_selection = 0
                elif self.menu_selection == 2:  # Model Select
                    self.state = GameState.MODEL_SELECT
                    self.menu_selection = 0
                elif self.menu_selection == 3:  # Quit
                    self.running = False
    
    def process_settings_input(self, event):
        """Process input for the settings screen."""
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                self.save_settings()
                self.state = GameState.MAIN_MENU
                self.menu_selection = 0
                self.play_sound('click')
            elif event.key == K_UP:
                self.menu_selection = (self.menu_selection - 1) % 12  # Updated number of options
                self.play_sound('click')
            elif event.key == K_DOWN:
                self.menu_selection = (self.menu_selection + 1) % 12  # Updated number of options
                self.play_sound('click')
            elif event.key == K_RETURN:
                self.play_sound('click')
                if self.menu_selection == 11:  # Save and Return (updated index)
                    self.save_settings()
                    self.state = GameState.MAIN_MENU
                    self.menu_selection = 0
                elif self.menu_selection == 3:  # AI Only Mode toggle
                    self.settings['AI_ONLY_MODE'] = not self.settings['AI_ONLY_MODE']
                elif self.menu_selection == 5:  # Show Probabilities toggle
                    self.settings['SHOW_PROBABILITIES'] = not self.settings['SHOW_PROBABILITIES']
                elif self.menu_selection == 8:  # Sound Effects toggle
                    self.settings['SOUND_EFFECTS'] = not self.settings['SOUND_EFFECTS']
            # Change values with left/right arrow keys or space
            elif event.key in (K_LEFT, K_RIGHT, K_SPACE):
                self.play_sound('click')
                if self.menu_selection == 0:  # Buyin
                    change = -100 if event.key == K_LEFT else 100
                    self.settings['BUYIN'] = max(100, self.settings['BUYIN'] + change)
                elif self.menu_selection == 1:  # Small Blind
                    change = -1 if event.key == K_LEFT else 1
                    self.settings['SMALL_BLIND'] = max(1, self.settings['SMALL_BLIND'] + change)
                elif self.menu_selection == 2:  # Big Blind
                    change = -2 if event.key == K_LEFT else 2
                    self.settings['BIG_BLIND'] = max(2, self.settings['BIG_BLIND'] + change)
                elif self.menu_selection == 3:  # AI Only Mode toggle
                    self.settings['AI_ONLY_MODE'] = not self.settings['AI_ONLY_MODE']
                elif self.menu_selection == 5:  # Show Probabilities toggle
                    self.settings['SHOW_PROBABILITIES'] = not self.settings['SHOW_PROBABILITIES']
                elif self.menu_selection == 6:  # Probability Simulations
                    change = -1000 if event.key == K_LEFT else 1000
                    self.settings['PROBABILITY_SIMULATIONS'] = max(1000, self.settings['PROBABILITY_SIMULATIONS'] + change)
                elif self.menu_selection == 7:  # AI Temperature
                    change = -0.1 if event.key == K_LEFT else 0.1
                    self.settings['AI_TEMPERATURE'] = round(max(0.1, min(2.0, self.settings['AI_TEMPERATURE'] + change)), 1)
                elif self.menu_selection == 8:  # Sound Effects toggle
                    self.settings['SOUND_EFFECTS'] = not self.settings['SOUND_EFFECTS']
                elif self.menu_selection == 9:  # AI Think Delay
                    change = -0.1 if event.key == K_LEFT else 0.1
                    self.settings['AI_THINK_DELAY'] = round(max(0, min(3.0, self.settings['AI_THINK_DELAY'] + change)), 1)
                elif self.menu_selection == 10:  # Animation Speed
                    change = -1 if event.key == K_LEFT else 1
                    self.settings['ANIMATION_SPEED'] = max(1, min(10, self.settings['ANIMATION_SPEED'] + change))
            # Handle text input for player name
            elif self.menu_selection == 4:  # Human Name
                if event.key == K_BACKSPACE:
                    self.settings['HUMAN_PLAYER_NAME'] = self.settings['HUMAN_PLAYER_NAME'][:-1]
                elif event.unicode.isalnum() and len(self.settings['HUMAN_PLAYER_NAME']) < 10:
                    self.settings['HUMAN_PLAYER_NAME'] += event.unicode
    
    def process_model_select_input(self, event):
        """Process input for the AI model selection screen."""
        all_models = list(DEFAULT_AI_MODEL_SHORT_NAMES.keys())
        
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE or event.key == K_RETURN:
                self.save_active_models()
                self.state = GameState.MAIN_MENU
                self.menu_selection = 0
                self.play_sound('click')
            elif event.key == K_UP:
                self.menu_selection = max(0, self.menu_selection - 1)
                self.play_sound('click')
            elif event.key == K_DOWN:
                self.menu_selection = min(len(all_models) - 1, self.menu_selection + 1)
                self.play_sound('click')
            elif event.key == K_SPACE:
                # Toggle selection of current model
                current_model = all_models[self.menu_selection]
                if current_model in self.active_models:
                    # Ensure we always have at least one model
                    if len(self.active_models) > 1:
                        self.active_models.remove(current_model)
                else:
                    self.active_models.append(current_model)
                self.play_sound('click')
    
    def process_game_input(self, event):
        """Process input during gameplay."""
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                self.state = GameState.PAUSED
                self.play_sound('click')
            elif event.key == K_h:
                self.show_help = not self.show_help
                self.play_sound('click')
        
        # Handle mouse clicks on action buttons
        elif event.type == MOUSEBUTTONDOWN and event.button == 1:  # Left click
            mouse_pos = pygame.mouse.get_pos()
            
            for action, rect in self.action_buttons:
                if rect.collidepoint(mouse_pos):
                    self.play_sound('click')
                    
                    if action == 'FOLD':
                        if self.table._current_action_player_id == self.settings['HUMAN_PLAYER_ID']:
                            self.table.publicIn(self.settings['HUMAN_PLAYER_ID'], RoundPublicInId.FOLD)
                    
                    elif action == 'CHECK':
                        if self.table._current_action_player_id == self.settings['HUMAN_PLAYER_ID']:
                            self.table.publicIn(self.settings['HUMAN_PLAYER_ID'], RoundPublicInId.CHECK)
                    
                    elif action.startswith('CALL'):
                        if self.table._current_action_player_id == self.settings['HUMAN_PLAYER_ID']:
                            self.table.publicIn(self.settings['HUMAN_PLAYER_ID'], RoundPublicInId.CALL)
                    
                    elif action == 'RAISE':
                        # Activate the raise slider
                        self.slider_active = True
                        self.slider_value = 0  # Start at minimum
                    
                    elif action == 'CONFIRM_RAISE':
                        if self.table._current_action_player_id == self.settings['HUMAN_PLAYER_ID']:
                            self.table.publicIn(self.settings['HUMAN_PLAYER_ID'], RoundPublicInId.RAISE, 
                                              raise_by=self.raise_amount)
                            self.slider_active = False
                    
                    elif action == 'CANCEL_RAISE':
                        self.slider_active = False
        
        # Handle slider dragging
        elif self.slider_active:
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                slider_x = (SCREEN_WIDTH - 400) // 2
                slider_y = SCREEN_HEIGHT - 120
                slider_width = 400
                slider_height = 30
                
                # Check if click is on slider
                slider_rect = pygame.Rect(slider_x, slider_y, slider_width, slider_height)
                if slider_rect.collidepoint(mouse_pos):
                    # Update slider value based on click position
                    self.slider_value = ((mouse_pos[0] - slider_x) / slider_width) * 100
                    self.slider_value = max(0, min(100, self.slider_value))
            
            elif event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0]:  # Left button held
                mouse_pos = pygame.mouse.get_pos()
                slider_x = (SCREEN_WIDTH - 400) // 2
                slider_width = 400
                
                # Update slider value based on drag position
                if mouse_pos[0] >= slider_x and mouse_pos[0] <= slider_x + slider_width:
                    self.slider_value = ((mouse_pos[0] - slider_x) / slider_width) * 100
                    self.slider_value = max(0, min(100, self.slider_value))
    
    def process_round_end_input(self, event):
        """Process input at the end of a round."""
        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                # Clear old round info before starting a new one
                self.play_sound('click')
                
                # Reset round state 
                if hasattr(self.table, 'round'):
                    self.table.round = None  # Clear old round object
                
                # Reset flags
                if hasattr(self.table, 'round_over_flag'):
                    self.table.round_over_flag = False
                self.table.pending_winner_messages.clear()
                self.table.pending_showdown_messages.clear()
                self.message_log.append("Starting new round...")
                
                # Start a new round
                active_players = self.table.seats.getPlayerGroup()
                if len(active_players) >= 2:
                    self.state = GameState.PLAYING  # Change state before starting round
                    self.start_round()
                else:
                    self.message_log.append("Not enough players to continue!")
                    self.state = GameState.MAIN_MENU
                    self.menu_selection = 0
            
            elif event.key == K_RETURN:
                # End the game and show final standings
                self.play_sound('click')
                
                # Draw game over screen with final standings
                self.screen.fill(POKER_DARK_GREEN)
                
                # Title
                title_text = "GAME OVER - FINAL STANDINGS"
                title_surf, title_rect = self.title_font.render(title_text, GOLD)
                title_rect.center = (self.screen_width // 2, 100)
                self.screen.blit(title_surf, title_rect)
                
                # Show player standings sorted by money
                active_players = self.table.seats.getPlayerGroup()
                if active_players:
                    # Sort players by money
                    sorted_players = sorted(active_players, 
                                          key=lambda p: p.money if hasattr(p, 'money') else 0, 
                                          reverse=True)
                    
                    # Draw standings table
                    header_text = "PLAYER           FINAL STACK"
                    header_surf, header_rect = self.menu_font.render(header_text, WHITE)
                    header_rect.topleft = (self.screen_width // 4, 200)
                    self.screen.blit(header_surf, header_rect)
                    
                    # List players
                    for i, player in enumerate(sorted_players):
                        player_name = player.name if hasattr(player, 'name') else f"Player {player.id}"
                        player_money = player.money if hasattr(player, 'money') else 0
                        
                        # Highlight the winner
                        color = GOLD if i == 0 else WHITE
                        
                        # Format the row
                        row_text = f"{player_name.ljust(15)} ${player_money}"
                        row_surf, row_rect = self.game_font.render(row_text, color)
                        row_rect.topleft = (self.screen_width // 4, 250 + i * 40)
                        self.screen.blit(row_surf, row_rect)
                
                # Return to menu text
                menu_text = "PRESS ANY KEY TO RETURN TO MAIN MENU"
                menu_surf, menu_rect = self.game_font.render(menu_text, CYAN)
                menu_rect.center = (self.screen_width // 2, self.screen_height - 100)
                self.screen.blit(menu_surf, menu_rect)
                
                # Render and wait for key
                pygame.display.flip()
                
                # Wait for a key press to return to menu
                waiting_for_key = True
                while waiting_for_key:
                    for evt in pygame.event.get():
                        if evt.type == KEYDOWN or evt.type == QUIT:
                            waiting_for_key = False
                            break
                    self.clock.tick(30)
                
                # Return to main menu
                self.state = GameState.MAIN_MENU
                self.menu_selection = 0
                
            elif event.key == K_ESCAPE:
                self.state = GameState.MAIN_MENU
                self.menu_selection = 0
                self.play_sound('click')
    
    def process_paused_input(self, event):
        """Process input when the game is paused."""
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                self.state = GameState.PLAYING
                self.play_sound('click')
            elif event.key == K_m:
                # Return to main menu
                self.state = GameState.MAIN_MENU
                self.menu_selection = 0
                self.play_sound('click')
    
    def update_game(self):
        """Update the game state each frame."""
        if self.state == GameState.PLAYING:
            if hasattr(self.table, 'round_over_flag') and self.table.round_over_flag:
                self.state = GameState.ROUND_END
                return
            
            # Check for player action required
            if hasattr(self.table, '_current_action_player_id') and self.table._current_action_player_id:
                action_player_id = self.table._current_action_player_id
                
                # If AI player's turn, get their action
                is_ai_turn = self.settings['AI_ONLY_MODE'] or action_player_id != self.settings['HUMAN_PLAYER_ID']
                if is_ai_turn:
                    # Add a message to the log
                    player_name = self.table._get_player_name(action_player_id)
                    self.message_log.append(f"{player_name} is thinking...")
                    
                    # Use the configured AI think delay from settings
                    ai_delay = self.settings.get('AI_THINK_DELAY', 0.5)
                    # Convert to milliseconds for pygame.time.delay
                    delay_ms = int(ai_delay * 1000) if ai_delay > 0 else 0
                    
                    # Apply the delay
                    if delay_ms > 0:
                        pygame.time.delay(delay_ms)
                    
                    # Get AI action using the imported function
                    action_enum, action_kwargs = get_ai_action(self.table, action_player_id)
                    
                    # Execute the action
                    self.table.publicIn(action_player_id, action_enum, **action_kwargs)
                    
                    # Log the action
                    action_str = action_enum.name
                    if action_enum == RoundPublicInId.RAISE:
                        action_str += f" {action_kwargs['raise_by']}"
                    self.message_log.append(f"{player_name}: {action_str}")
                    
                    # Play sound for AI action
                    if action_enum == RoundPublicInId.FOLD:
                        self.play_sound('click')
                    else:
                        self.play_sound('chips')
    
    def update(self):
        """Main update method for the game."""
        if self.state == GameState.PLAYING or self.state == GameState.ROUND_END:
            self.update_game()
    
    def render(self):
        """Render the current game state to the screen."""
        if self.state == GameState.MAIN_MENU:
            self.draw_menu()
        elif self.state == GameState.SETTINGS:
            self.draw_settings()
        elif self.state == GameState.MODEL_SELECT:
            self.draw_model_select()
        elif self.state == GameState.PLAYING:
            self.draw_game()
        elif self.state == GameState.ROUND_END:
            self.draw_round_end()
        elif self.state == GameState.PAUSED:
            # First draw the game state
            self.draw_game()
            
            # Then overlay with pause menu
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            pause_text = "GAME PAUSED"
            pause_surf, pause_rect = self.title_font.render(pause_text, WHITE)
            pause_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
            self.screen.blit(pause_surf, pause_rect)
            
            continue_text = "PRESS ESC TO CONTINUE"
            continue_surf, continue_rect = self.menu_font.render(continue_text, WHITE)
            continue_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20)
            self.screen.blit(continue_surf, continue_rect)
            
            menu_text = "PRESS M FOR MAIN MENU"
            menu_surf, menu_rect = self.menu_font.render(menu_text, WHITE)
            menu_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 70)
            self.screen.blit(menu_surf, menu_rect)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                
                # Process input based on current state
                if self.state == GameState.MAIN_MENU:
                    self.process_menu_input(event)
                elif self.state == GameState.SETTINGS:
                    self.process_settings_input(event)
                elif self.state == GameState.MODEL_SELECT:
                    self.process_model_select_input(event)
                elif self.state == GameState.PLAYING:
                    self.process_game_input(event)
                elif self.state == GameState.ROUND_END:
                    self.process_round_end_input(event)
                elif self.state == GameState.PAUSED:
                    self.process_paused_input(event)
            
            # Update game state
            self.update()
            
            # Render the current frame
            self.render()
            
            # Cap the frame rate
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

class PyGameTable(Table):
    """
    Custom Table class extending pokerlib's Table to handle PyGame interaction,
    AI player management, game state display, probability calculation, and specific event handling.
    This is adapted from the CommandLineTable class from the original game.py.
    """
    def __init__(self, *args, **kwargs):
        """Initializes the PyGameTable."""
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
        self.current_betting_round_actions = []  # Store actions for current betting round

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

    # The rest of the methods should be imported from the original CommandLineTable class
    # For brevity, I'm including just the assign_ai_models method, which is needed for initialization
    
    def assign_ai_models(self):
        """Assigns AI models from the active models list to AI players."""
        # Use a direct approach for model assignment
        
        ai_player_ids = []
        player_group = self.seats.getPlayerGroup()
        if not player_group: return  # No players to assign to
        
        # Get active models from globals or default
        # In a real implementation, this should be passed from the PokerGame instance
        active_models = DEFAULT_AI_MODELS
        
        # Identify AI player IDs (all players in AI_ONLY_MODE, or non-human players)
        for p in player_group:
            if p and hasattr(p, 'id'):
                ai_player_ids.append(p.id)
        
        num_ai = len(ai_player_ids)
        if num_ai == 0 or not active_models:
            print("No AI players or no AI models defined. Skipping model assignment.")
            return  # No AI players or no models defined
        
        # Clear previous assignments
        self.ai_model_assignments.clear()
        
        # Assign models cyclically
        for i, pid in enumerate(ai_player_ids):
            model_full_name = active_models[i % len(active_models)]  # Cycle through models
            self.ai_model_assignments[pid] = model_full_name  # Store the full name
            
            # Get short name for logging/display
            model_short_name = DEFAULT_AI_MODEL_SHORT_NAMES.get(model_full_name, model_full_name.split('/')[-1])
            player_name = self._get_player_name(pid)
            print(f"AI Player {player_name} assigned model: {model_short_name}")
    
    def _get_player_name(self, player_id):
        """Safely retrieves a player's name by their ID."""
        player = self.seats.getPlayerById(player_id)
        return player.name if player and hasattr(player, 'name') else f"P{player_id}"

    def publicOut(self, out_id, **kwargs):
        """Handles public events broadcast by the poker engine."""
        # Implementation based on the original CommandLineTable.publicOut
        
        player_id = kwargs.get("player_id")
        
        # Check if a specific type of event
        is_round_event = isinstance(out_id, RoundPublicOutId)
        is_table_event = isinstance(out_id, TablePublicOutId)
        
        # --- Round Events ---
        if is_round_event:
            if out_id == RoundPublicOutId.PLAYERACTIONREQUIRED:
                self._current_action_player_id = player_id
                self.to_call = kwargs.get('to_call', 0)
                self.can_check = self.to_call == 0
                
                player = self.seats.getPlayerById(player_id)
                self.can_raise = player and hasattr(player, 'money') and player.money > self.to_call
                self.min_raise = self.big_blind
                
                # Calculate probabilities for the first player action in each round
                current_turn_idx = self.round.turn.value if self.round and hasattr(self.round, 'turn') else -1
                if current_turn_idx > self.last_turn_processed:
                    # This would be where we'd calculate probabilities
                    # self._calculate_win_probabilities()
                    self.last_turn_processed = current_turn_idx
            
            elif out_id == RoundPublicOutId.NEWTURN:
                # Clear actions for the new betting round
                self.current_betting_round_actions.clear()
            
            elif out_id == RoundPublicOutId.PLAYERCHECK:
                player_name = self._get_player_name(player_id)
                action_msg = f"{player_name} checks"
                self.current_betting_round_actions.append(action_msg)
            
            elif out_id == RoundPublicOutId.PLAYERCALL:
                player_name = self._get_player_name(player_id)
                action_msg = f"{player_name} calls ${kwargs['paid_amount']}"
                self.current_betting_round_actions.append(action_msg)
            
            elif out_id == RoundPublicOutId.PLAYERFOLD:
                player_name = self._get_player_name(player_id)
                action_msg = f"{player_name} folds"
                self.current_betting_round_actions.append(action_msg)
            
            elif out_id == RoundPublicOutId.PLAYERRAISE:
                player_name = self._get_player_name(player_id)
                action_msg = f"{player_name} raises by ${kwargs['raised_by']} (total bet this street: ${kwargs['paid_amount']})"
                self.current_betting_round_actions.append(action_msg)
            
            elif out_id == RoundPublicOutId.PLAYERWENTALLIN:
                player_name = self._get_player_name(player_id)
                action_msg = f"{player_name} goes ALL-IN with ${kwargs['paid_amount']}!"
                self.current_betting_round_actions.append(action_msg)
            
            elif out_id == RoundPublicOutId.ROUNDFINISHED:
                self.round_over_flag = True
            
            elif out_id == RoundPublicOutId.DECLAREPREMATUREWINNER:
                player_name = self._get_player_name(player_id)
                winner_msg = f"{player_name} wins ${kwargs['money_won']} (Prematurely - all others folded)"
                self.pending_winner_messages.append(winner_msg)
            
            elif out_id == RoundPublicOutId.DECLAREFINISHEDWINNER:
                player_name = self._get_player_name(player_id)
                hand_name = kwargs.get('handname')
                hand_name_str = hand_name.name.replace("_", " ").title() if hand_name else "Unknown Hand"
                winner_msg = f"{player_name} wins ${kwargs['money_won']} with {hand_name_str}"
                self.pending_winner_messages.append(winner_msg)
            
            elif out_id == RoundPublicOutId.PUBLICCARDSHOW:
                player = self.seats.getPlayerById(player_id)
                shown_cards_raw = kwargs.get('cards', [])
                shown_cards_tuples = [tuple(c) for c in shown_cards_raw if isinstance(c, (list, tuple)) and len(c)==2]
                
                if player and shown_cards_tuples:
                    self._player_cards[player.id] = tuple(shown_cards_tuples)
                    self.showdown_occurred = True
        
        # --- Table Events ---
        elif is_table_event:
            if out_id == TablePublicOutId.PLAYERJOINED:
                pass  # Handle player joining
            
            elif out_id == TablePublicOutId.PLAYERREMOVED:
                # Clean up player data when they leave
                if player_id in self._player_cards: 
                    del self._player_cards[player_id]
                if player_id in self.ai_model_assignments: 
                    del self.ai_model_assignments[player_id]
                if player_id in self._player_probabilities: 
                    del self._player_probabilities[player_id]
    
    def _newRound(self, round_id):
        """Handles the initialization of a new round."""
        print(f"Initializing round {round_id}")
        
        current_players = self.seats.getPlayerGroup()
        
        if not current_players:
            print(f"Error: Cannot start round {round_id} - No players at the table.")
            self.round = None  # Ensure round object is None
            return
        
        # Reset player states if method exists
        for player in current_players:
            if hasattr(player, "resetState"):
                try: 
                    player.resetState()
                except Exception as e: 
                    print(f"Could not reset state for player {player.id}: {e}")
        
        # Reset round-specific state
        self._player_cards.clear()  # Clear stored hole cards from previous round
        self.pending_winner_messages.clear()
        self.pending_showdown_messages.clear()
        self.printed_showdown_board = False
        self.showdown_occurred = False
        self.round_over_flag = False  # Reset round over flag
        self._current_action_player_id = None  # Clear any lingering action request
        self.to_call = 0
        self.can_check = False
        self.can_raise = False
        self.min_raise = self.big_blind  # Reset min raise to default
        self._player_probabilities.clear()  # Clear probabilities for the new round
        self.last_turn_processed = -1  # Reset turn tracking for probability calc
        self.current_betting_round_actions.clear()  # Clear actions for the new round
        
        # Clear AI message history for the new round ID if it exists
        if round_id in self.ai_message_history:
            self.ai_message_history[round_id].clear()
        
        # Assign AI models if needed
        if not self.ai_model_assignments:
            self.assign_ai_models()
        
        # Initialize the pokerlib Round
        try:
            self.round = self.RoundClass(
                round_id,
                current_players,
                self.button,
                self.small_blind,
                self.big_blind
            )
            print(f"Successfully initialized Round {round_id}.")
        except Exception as e:
            print(f"Failed to initialize Round {round_id}: {e}")
            self.round = None  # Ensure round object is None if init fails
    
    def privateOut(self, player_id, out_id, **kwargs):
        """Handles private events sent to a specific player."""
        if isinstance(out_id, RoundPrivateOutId):
            if out_id == RoundPrivateOutId.DEALTCARDS:
                cards_raw = kwargs.get('cards')
                if cards_raw and isinstance(cards_raw, (list, tuple)) and len(cards_raw) == 2:
                    cards_tuples = tuple(tuple(c) for c in cards_raw if isinstance(c, (list, tuple)) and len(c)==2)
                    if len(cards_tuples) == 2:
                        self._player_cards[player_id] = cards_tuples
                        print(f"Dealt cards to player {self._get_player_name(player_id)}")
        
        elif isinstance(out_id, TablePrivateOutId):
            if out_id == TablePrivateOutId.BUYINTOOLOW:
                print(f"Error: Buy-in too low for player {player_id}")
            elif out_id == TablePrivateOutId.TABLEFULL:
                print(f"Error: Table is full when player {player_id} tried to join")
            elif out_id == TablePrivateOutId.PLAYERALREADYATTABLE:
                print(f"Error: Player {player_id} is already at the table")
            elif out_id == TablePrivateOutId.PLAYERNOTATTABLE:
                print(f"Error: Player {player_id} is not at the table")
            elif out_id == TablePrivateOutId.INCORRECTSEATINDEX:
                print(f"Error: Incorrect seat index for player {player_id}")

# Only run the game if executed directly
if __name__ == "__main__":
    game = PokerGame()
    game.run()