import sys
import time      # For sleep
import traceback # For better error reporting
import os        # For screen clearing
from collections import deque

# Necessary imports from pokerlib
from pokerlib import Player, PlayerSeats, Table
from pokerlib.enums import ( Hand, Rank, RoundPrivateOutId, RoundPublicInId,
                            RoundPublicOutId, Suit, TablePrivateOutId,
                            TablePublicInId, TablePublicOutId, Turn )

# --- Configuration ---
CLEAR_SCREEN = False # Set to False if screen clearing causes issues

# --- ANSI Color Codes ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLACK = "\033[30m"; RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"; BRIGHT_RED = "\033[91m"; BRIGHT_GREEN = "\033[92m"; BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"; BRIGHT_MAGENTA = "\033[95m"; BRIGHT_CYAN = "\033[96m"; BRIGHT_WHITE = "\033[97m"
    BG_BLACK = "\033[40m"; BG_RED = "\033[41m"; BG_GREEN = "\033[42m"; BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"; BG_MAGENTA = "\033[45m"; BG_CYAN = "\033[46m"; BG_WHITE = "\033[47m"

def colorize(text, color):
    """Applies ANSI color code and resets."""
    return f"{color}{text}{Colors.RESET}"

# --- Helper Functions for Display (Enhanced) ---

def format_card(card):
    """Formats a card tuple (Rank, Suit) into a short, colored string."""
    if not card: return colorize("??", Colors.BRIGHT_BLACK)
    rank_map = { Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5", Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9", Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A" }
    suit_map = {Suit.SPADE: "♠", Suit.CLUB: "♣", Suit.DIAMOND: "♦", Suit.HEART: "♥"}
    suit_color_map = { Suit.SPADE: Colors.WHITE, Suit.CLUB: Colors.WHITE, Suit.DIAMOND: Colors.BRIGHT_RED, Suit.HEART: Colors.BRIGHT_RED }
    try:
        # Ensure input is treated as a sequence (tuple or list) and has 2 elements
        if hasattr(card, '__len__') and len(card) == 2:
             rank, suit = card
             rank_str = rank_map[rank]; suit_str = suit_map[suit]
             color = suit_color_map.get(suit, Colors.WHITE)
             card_text = f"{rank_str}{suit_str}"
             if rank >= Rank.JACK: card_text = Colors.BOLD + card_text # Bold face cards
             return colorize(card_text, color)
        else: return colorize("??", Colors.BRIGHT_BLACK) # Invalid card format
    except (TypeError, KeyError, ValueError, IndexError): return colorize("??", Colors.BRIGHT_BLACK)

def format_cards(cards):
    """Formats a list/tuple of cards with colors."""
    # Ensure cards is iterable before processing
    if hasattr(cards, '__iter__'):
        return " ".join(format_card(c) for c in cards)
    return ""

def format_hand_enum(hand_enum):
    """Formats a Hand enum member into a readable string."""
    return hand_enum.name.replace("_", " ").title() if hand_enum else "Unknown Hand"

def clear_terminal():
    """Clears the terminal screen."""
    if CLEAR_SCREEN: os.system('cls' if os.name == 'nt' else 'clear')

# --- Custom Table Class for IO (Enhanced Display) ---

class CommandLineTable(Table):
    """Overrides Table to handle command-line input and output with enhanced visuals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._player_cards = {} # Stores tuple card representation: {player_id: ((R,S),(R,S))}
        self._current_action_player_id = None
        self.round_over_flag = False

    def _get_player_name(self, player_id):
        player = self.seats.getPlayerById(player_id)
        return player.name if player else f"Player {player_id}"

    # --- OVERRIDE _newRound ---
    def _newRound(self, round_id):
        current_players = self.seats.getPlayerGroup()
        if not current_players: print(colorize("Error: No players.", Colors.RED)); self.round = None; return
        for player in current_players:
            if hasattr(player, "resetState"): player.resetState()
        self._player_cards.clear()
        try:
            self.round = self.RoundClass( round_id, current_players, self.button, self.small_blind, self.big_blind )
            self.round_over_flag = False
        except Exception as e:
            print(colorize(f"--- ROUND INIT ERROR (Round {round_id}) ---", Colors.RED)); traceback.print_exc()
            self.round = None

    # --- END OVERRIDE ---

    def _display_game_state(self):
        """ Displays the current state of the table and round with enhanced visuals. """
        clear_terminal()
        title = colorize("====== POKER GAME STATE ======", Colors.BRIGHT_CYAN + Colors.BOLD)
        separator = colorize("--------------------------------------------------", Colors.BRIGHT_BLACK)
        print(f"\n{title}")

        if ( not self.round or not hasattr(self.round, 'id') or not hasattr(self.round, 'players')
             or not isinstance(self.round.players, list) or not hasattr(self.round, 'turn')
             or not hasattr(self.round, 'board') or not hasattr(self.round, 'pot_size')
             or not isinstance(self.round.pot_size, list)):
            print(colorize("No active round or essential round state missing.", Colors.YELLOW))
            print(colorize("\nPlayers at table:", Colors.YELLOW))
            for player in self.seats.getPlayerGroup():
                money_str = f"${player.money}" if hasattr(player, 'money') else colorize("N/A", Colors.BRIGHT_BLACK)
                print(f"  - {colorize(player.name, Colors.CYAN)}: {colorize(money_str, Colors.BRIGHT_GREEN)}")
            print(separator); return

        try: # Safely access round attributes
            round_id = self.round.id; turn_name = self.round.turn.name
            board_cards_list = [tuple(c) for c in self.round.board if isinstance(c, list) and len(c)==2] # Ensure tuples
            board_cards_str = format_cards(board_cards_list)
            pot_total = sum(self.round.pot_size); current_turn_enum = self.round.turn
            players_in_round = self.round.players
            button_idx = self.round.button if hasattr(self.round, "button") else -1
            sb_idx = self.round.small_blind_player_index if hasattr(self.round, "small_blind_player_index") else -1
            bb_idx = self.round.big_blind_player_index if hasattr(self.round, "big_blind_player_index") else -1
        except (AttributeError, TypeError, IndexError) as e:
            print(colorize(f"Error accessing round details for display: {e}", Colors.BRIGHT_RED)); return

        print(f"Round: {colorize(str(round_id), Colors.WHITE)}   Turn: {colorize(turn_name, Colors.WHITE + Colors.BOLD)}")
        print(f"Board: [ {board_cards_str} ]")
        print(f"Pot:   {colorize(f'${pot_total}', Colors.BRIGHT_YELLOW + Colors.BOLD)}")
        print(colorize("\nPlayers:", Colors.YELLOW))

        max_name_len = 0
        if players_in_round: max_name_len = max(len(p.name) for p in players_in_round if p and hasattr(p, 'name'))

        for idx, player in enumerate(players_in_round):
            if not player or not hasattr(player, "id"): continue

            is_acting = (player.id == self._current_action_player_id)
            line_prefix = colorize(" > ", Colors.BRIGHT_YELLOW + Colors.BOLD) if is_acting else "   "
            player_name_str = (player.name if hasattr(player, 'name') else f'P{player.id}').ljust(max_name_len)
            player_name_colored = colorize(player_name_str, Colors.CYAN + (Colors.BOLD if is_acting else ""))
            money_val = player.money if hasattr(player, 'money') else 0
            money_str = colorize(f"${money_val}", Colors.BRIGHT_GREEN)

            cards_str = colorize("( ? ? )", Colors.BRIGHT_BLACK)
            if player.id in self._player_cards: # Use the stored tuple cards
                 cards_str = f"( {format_cards(self._player_cards[player.id])} )"
            # Check player.cards for showdown display (ensure it contains tuples)
            elif (self.round_over_flag or self.round.finished) and not player.is_folded and hasattr(player, "cards"):
                 cards_data = player.cards
                 if isinstance(cards_data, (list, tuple)): # Check if iterable
                     cards_tuple_list = [tuple(c) for c in cards_data if isinstance(c, (list, tuple)) and len(c)==2]
                     if len(cards_tuple_list) == 2: # Ensure conversion worked
                          cards_str = f"( {format_cards(cards_tuple_list)} )"

            status = []; player_round_index = idx
            if hasattr(player, 'is_folded') and player.is_folded: status.append(colorize("FOLDED", Colors.BRIGHT_BLACK))
            if hasattr(player, 'is_all_in') and player.is_all_in: status.append(colorize("ALL-IN", Colors.BRIGHT_RED + Colors.BOLD))
            if player_round_index == button_idx: status.append(colorize("D", Colors.WHITE + Colors.BOLD))
            if player_round_index == sb_idx: status.append(colorize("SB", Colors.YELLOW))
            if player_round_index == bb_idx: status.append(colorize("BB", Colors.YELLOW))
            status_str = " ".join(status)

            turn_stake_val = 0
            if ( hasattr(player, "turn_stake") and isinstance(player.turn_stake, list) and len(player.turn_stake) > current_turn_enum.value ):
                turn_stake_val = player.turn_stake[current_turn_enum.value]
            stake_str = colorize(f"[Bet: ${turn_stake_val}]", Colors.MAGENTA) if turn_stake_val > 0 else ""

            print(f"{line_prefix}{player_name_colored} {money_str.ljust(8)} {cards_str.ljust(20)} {stake_str.ljust(15)} {status_str}")

        print(separator)

    # --- Output Handling (Enhanced) ---
    def publicOut(self, out_id, **kwargs):
        player_id = kwargs.get("player_id")
        player_name_raw = self._get_player_name(player_id) if player_id else "System"
        player_name = colorize(player_name_raw, Colors.CYAN)
        msg = ""; prefix = ""

        self._current_action_player_id = None
        processed = False

        if isinstance(out_id, RoundPublicOutId):
            processed = True; player = self.seats.getPlayerById(player_id) if player_id else None
            prefix = colorize("[ROUND]", Colors.BLUE)
            if out_id == RoundPublicOutId.NEWROUND: msg = "Dealing cards..."
            elif out_id == RoundPublicOutId.NEWTURN:
                 prefix = "" # Displayed via _display_game_state
                 if hasattr(self.round, 'board'):
                     self.community_cards = [tuple(card) for card in self.round.board if isinstance(card, list) and len(card)==2]
                 self._display_game_state()
            elif out_id == RoundPublicOutId.SMALLBLIND: msg = f"{player_name} posts {colorize('Small Blind', Colors.YELLOW)} ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.BIGBLIND: msg = f"{player_name} posts {colorize('Big Blind', Colors.YELLOW)} ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.PLAYERCHECK: prefix = colorize("[ACTION]", Colors.GREEN); msg = f"{player_name} checks"
            elif out_id == RoundPublicOutId.PLAYERCALL: prefix = colorize("[ACTION]", Colors.GREEN); msg = f"{player_name} calls ${kwargs['paid_amount']}"
            elif out_id == RoundPublicOutId.PLAYERFOLD: prefix = colorize("[ACTION]", Colors.BRIGHT_BLACK); msg = f"{player_name} folds"
            elif out_id == RoundPublicOutId.PLAYERRAISE: prefix = colorize("[ACTION]", Colors.BRIGHT_MAGENTA); msg = f"{player_name} raises by ${kwargs['raised_by']} (bets ${kwargs['paid_amount']})"
            elif out_id == RoundPublicOutId.PLAYERISALLIN: prefix = colorize("[INFO]", Colors.BRIGHT_RED); msg = f"{player_name} is {colorize('ALL-IN!', Colors.BOLD)}"
            elif out_id == RoundPublicOutId.PLAYERWENTALLIN: prefix = colorize("[ACTION]", Colors.BRIGHT_RED + Colors.BOLD); msg = f"{player_name} goes ALL-IN with ${kwargs['paid_amount']}!"
            elif out_id == RoundPublicOutId.PLAYERACTIONREQUIRED:
                prefix = "" # Handled by get_player_action prompt
                self._current_action_player_id = player_id
            elif out_id == RoundPublicOutId.PLAYERCHOICEREQUIRED: pass # Ignoring
            elif out_id == RoundPublicOutId.PUBLICCARDSHOW:
                prefix = colorize("[SHOWDOWN]", Colors.WHITE)
                shown_cards_raw = kwargs.get('cards', []);
                # Ensure cards are tuples for formatting and potential state storage
                shown_cards_tuples = [tuple(c) for c in shown_cards_raw if isinstance(c, (list, tuple)) and len(c)==2]
                # Store shown cards in _player_cards map for display consistency
                if player and shown_cards_tuples: self._player_cards[player.id] = tuple(shown_cards_tuples)
                hand_name = format_hand_enum(kwargs.get('handenum'))
                msg = f"{player_name} shows {format_cards(shown_cards_tuples)} ({hand_name})"
            elif out_id == RoundPublicOutId.DECLAREPREMATUREWINNER: prefix = colorize("[WINNER]", Colors.BRIGHT_YELLOW + Colors.BOLD); msg = f"{player_name} wins ${kwargs['money_won']} (Premature)"
            elif out_id == RoundPublicOutId.DECLAREFINISHEDWINNER:
                 prefix = colorize("[WINNER]", Colors.BRIGHT_YELLOW + Colors.BOLD)
                 hand_name = format_hand_enum(kwargs.get('handname')); msg = f"{player_name} wins ${kwargs['money_won']} with {hand_name}"
            elif out_id == RoundPublicOutId.ROUNDFINISHED: prefix = ""; msg = colorize("\n======= ROUND FINISHED =======", Colors.BRIGHT_CYAN); self.round_over_flag = True; self._display_game_state()
            elif out_id == RoundPublicOutId.ROUNDCLOSED: prefix = colorize("[Internal]", Colors.BRIGHT_BLACK); msg = "Round Closed State."; self._player_cards.clear()

        elif isinstance(out_id, TablePublicOutId):
            processed = True; prefix = colorize("[TABLE]", Colors.MAGENTA)
            if out_id == TablePublicOutId.PLAYERJOINED: msg = f"{player_name} joined seat {kwargs['player_seat']}"
            elif out_id == TablePublicOutId.PLAYERREMOVED: msg = f"{player_name} left table"
            elif out_id == TablePublicOutId.NEWROUNDSTARTED: prefix = ""; msg = "" # Don't print, _newRound handles setup
            elif out_id == TablePublicOutId.ROUNDNOTINITIALIZED: prefix = colorize("[ERROR]", Colors.RED); msg = "No round running"
            elif out_id == TablePublicOutId.ROUNDINPROGRESS: prefix = colorize("[ERROR]", Colors.RED); msg = "Round already in progress"
            elif out_id == TablePublicOutId.INCORRECTNUMBEROFPLAYERS: prefix = colorize("[ERROR]", Colors.RED); msg = "Need 2+ players"

        # Print message if not handled by _display_game_state
        if msg and out_id != RoundPublicOutId.NEWTURN and out_id != RoundPublicOutId.ROUNDFINISHED:
             print(f"{prefix} {msg}")
        elif not processed and out_id != RoundPublicOutId.PLAYERCHOICEREQUIRED: # Print unhandled only if truly unknown
            print(colorize(f"Unhandled Public Out: ID={out_id} Data: {kwargs}", Colors.BRIGHT_BLACK))

    def privateOut(self, player_id, out_id, **kwargs):
        player_name_raw = self._get_player_name(player_id); player_name = colorize(player_name_raw, Colors.CYAN)
        prefix = colorize(f"[PRIVATE to {player_name_raw}]", Colors.YELLOW)
        msg = ""

        if out_id == RoundPrivateOutId.DEALTCARDS:
            cards_raw = kwargs.get('cards');
            if cards_raw and len(cards_raw) == 2:
                # Ensure inner elements are tuples before storing
                cards_tuples = tuple(tuple(c) for c in cards_raw if isinstance(c, (list, tuple)) and len(c)==2)
                if len(cards_tuples) == 2:
                     self._player_cards[player_id] = cards_tuples # Store tuple of tuples
                     msg = f"You are dealt {format_cards(cards_tuples)}"
                else: msg = colorize("Card conversion error.", Colors.RED)
            else: msg = colorize("Dealing error (no cards data).", Colors.RED)
        elif out_id == TablePrivateOutId.BUYINTOOLOW: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Buy-in of ${self.buyin} required."
        elif out_id == TablePrivateOutId.TABLEFULL: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Table full"
        elif out_id == TablePrivateOutId.PLAYERALREADYATTABLE: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Already seated"
        elif out_id == TablePrivateOutId.PLAYERNOTATTABLE: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Not at table"
        elif out_id == TablePrivateOutId.INCORRECTSEATINDEX: prefix = colorize(f"[ERROR to {player_name_raw}]", Colors.RED); msg = f"Bad seat index"
        else: prefix = colorize(f"[UNHANDLED PRIVATE to {player_name_raw}]", Colors.BRIGHT_BLACK); msg = f"ID={out_id} Data: {kwargs}"

        if msg: print(f"{prefix} {msg}")

# --- Main Game Logic (Enhanced Prompt) ---

def get_player_action(player_name, to_call, player_money, can_check, can_raise):
    """Prompts the user for an action and validates it with better visuals."""
    prompt_header = colorize(f"--- {player_name}'s Turn ---", Colors.BRIGHT_YELLOW + Colors.BOLD)
    print(prompt_header)
    actions = ["FOLD"]
    action_color_map = { "FOLD": Colors.BRIGHT_BLACK, "CHECK": Colors.BRIGHT_GREEN, "CALL": Colors.BRIGHT_CYAN, "RAISE": Colors.BRIGHT_MAGENTA }
    action_parts = [colorize("FOLD", action_color_map["FOLD"])]

    if can_check: actions.append("CHECK"); action_parts.append(colorize("CHECK", action_color_map["CHECK"]))
    elif to_call > 0 and player_money > 0:
        actions.append("CALL"); effective_call = min(to_call, player_money)
        action_parts.append(colorize(f"CALL({effective_call})", action_color_map["CALL"]))
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


# --- Main Execution ---

if __name__ == "__main__":
    NUM_PLAYERS = 3; BUYIN = 200; SMALL_BLIND = 5; BIG_BLIND = 10

    table = CommandLineTable( _id=0, seats=PlayerSeats([None] * NUM_PLAYERS), buyin=BUYIN, small_blind=SMALL_BLIND, big_blind=BIG_BLIND )

    players = []
    for i in range(NUM_PLAYERS):
        player_name = f"Player_{i+1}"; player = Player( table_id=table.id, _id=i + 1, name=player_name, money=BUYIN ); players.append(player)
        table.publicIn(player.id, TablePublicInId.BUYIN, player=player)

    clear_terminal()
    print(colorize("\n--- Welcome to Simple Pokerlib Game! ---", Colors.BRIGHT_CYAN + Colors.BOLD))
    print(f"{NUM_PLAYERS} players joined with {colorize(f'${BUYIN}', Colors.BRIGHT_GREEN)} each.")
    print(f"Blinds: {colorize(f'${SMALL_BLIND}/${BIG_BLIND}', Colors.YELLOW)}")

    round_count = 0
    try:
        while True:
            active_players_obj = table.seats.getPlayerGroup()
            if len(active_players_obj) < 2: print(colorize("\nNot enough players to continue. Game over.", Colors.YELLOW)); break
            round_count += 1
            initiator = active_players_obj[0] if active_players_obj else None
            if not initiator: print(colorize("Error: No players left to initiate round.", Colors.RED)); break
            table.publicIn( initiator.id, TablePublicInId.STARTROUND, round_id=round_count )

            # Inner loop for actions within a round
            while table.round and not table.round_over_flag:
                action_player_id_to_process = table._current_action_player_id
                if action_player_id_to_process:
                    table._current_action_player_id = None # Clear flag

                    player = table.seats.getPlayerById(action_player_id_to_process)
                    if not player: print(colorize(f"Warning: Action for missing player ID {action_player_id_to_process}", Colors.YELLOW)); continue

                    current_player_obj = None
                    if table.round and hasattr(table.round, "current_player"):
                        try: current_player_obj = table.round.current_player
                        except Exception: pass

                    if current_player_obj and player.id == current_player_obj.id:
                        if not all(hasattr(player, a) for a in ['money','stake','turn_stake']) or not isinstance(player.turn_stake, list) or not (table.round and hasattr(table.round, 'to_call')):
                             print(colorize(f"Warning: State not ready for {player.name}'s action.", Colors.YELLOW)); time.sleep(0.1); table._current_action_player_id = action_player_id_to_process; continue

                        to_call = table.round.to_call; can_check = to_call == 0; can_raise = player.money > to_call
                        action_enum, action_kwargs = get_player_action( player.name, to_call, player.money, can_check, can_raise )
                        table.publicIn(player.id, action_enum, **action_kwargs) # Send action

                    elif current_player_obj: # Log mismatch only if current player known
                        req_for = f"{player.name}({action_player_id_to_process})"; curr = f"{current_player_obj.name}({current_player_obj.id})"
                        print(colorize(f"Warning: Action request mismatch. Req for {req_for}, current is {curr}", Colors.YELLOW))
                time.sleep(0.05) # Reduce CPU usage slightly

            # <<< After inner loop >>>
            if table.round: table.round = None # Clear library's round object

            print(colorize("\nRound ended. Final stacks:", Colors.BRIGHT_WHITE))
            final_players = table.seats.getPlayerGroup()
            if not final_players: print("  No players remaining.")
            else:
                 for p in final_players:
                     money_val = p.money if hasattr(p, 'money') else 'N/A'
                     print(f"  - {colorize(p.name, Colors.CYAN)}: {colorize(f'${money_val}', Colors.BRIGHT_GREEN)}")

            try: # Ask to continue
                 cont = input(colorize("\nPlay another round? (y/n): ", Colors.WHITE)).lower()
                 if cont != 'y': break
            except EOFError: print(colorize("\nInput ended.", Colors.RED)); break

    except KeyboardInterrupt: print(colorize("\nCtrl+C detected. Exiting game.", Colors.YELLOW))
    except Exception as e:
         print(colorize("\n--- UNEXPECTED ERROR OCCURRED ---", Colors.BRIGHT_RED + Colors.BOLD)); traceback.print_exc()
         print(colorize("---------------------------------", Colors.BRIGHT_RED + Colors.BOLD))
    finally:
        print(colorize("\n--- Game Ended ---", Colors.BRIGHT_CYAN + Colors.BOLD)); print(colorize("Final Stacks:", Colors.WHITE))
        final_players = table.seats.getPlayerGroup()
        if not final_players: print("  No players remaining.")
        else:
            for p in final_players:
                money_str = f"${p.money}" if hasattr(p, 'money') else "N/A"
                print(f"  - {colorize(p.name, Colors.CYAN)}: {colorize(money_str, Colors.BRIGHT_GREEN)}")
        print(Colors.RESET) # Final color reset