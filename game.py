import sys
import time  # For sleep
import traceback  # For better error reporting
from collections import deque

# Necessary imports from pokerlib
from pokerlib import Player, PlayerSeats, Table
from pokerlib.enums import (Hand, Rank, RoundPrivateOutId, RoundPublicInId,
                            RoundPublicOutId, Suit, TablePrivateOutId,
                            TablePublicInId, TablePublicOutId, Turn)

# --- Helper Functions for Display ---


def format_card(card):
    """Formats a card tuple (Rank, Suit) into a short string like 'AS' or 'TH'."""
    if not card:
        return "??"
    rank_map = {
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
    suit_map = {Suit.SPADE: "s", Suit.CLUB: "c", Suit.DIAMOND: "d", Suit.HEART: "h"}
    try:
        rank, suit = card
        return rank_map[rank] + suit_map[suit]
    except (TypeError, KeyError, ValueError):
        return "??"


def format_cards(cards):
    """Formats a list/tuple of cards."""
    return " ".join(format_card(c) for c in cards) if cards else ""


def format_hand_enum(hand_enum):
    """Formats a Hand enum member into a readable string."""
    return hand_enum.name.replace("_", " ").title() if hand_enum else "Unknown Hand"


# --- Custom Table Class for IO ---


class CommandLineTable(Table):
    """Overrides Table to handle command-line input and output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._player_cards = {}
        self._current_action_player_id = None
        self.round_over_flag = False

    def _get_player_name(self, player_id):
        """Safely gets a player's name by ID."""
        player = self.seats.getPlayerById(player_id)
        return player.name if player else f"Player {player_id}"

    # --- OVERRIDE _newRound ---
    def _newRound(self, round_id):
        """
        Overrides the default _newRound to reset player states *before*
        the Round object is initialized, preventing state timing issues.
        """
        current_players = self.seats.getPlayerGroup()
        if not current_players:
            print("Error in _newRound: No players found to start round.")
            self.round = None
            return

        # print(f"[DEBUG] Resetting state for {len(current_players)} players before Round {round_id} init.") # Optional debug
        for player in current_players:
            if hasattr(player, "resetState"):
                player.resetState()
            else:  # Should not happen with standard Player object
                print(f"Warning: Player object {player} missing resetState method.")

        # Now call the original RoundClass constructor with the *reset* players
        try:
            # Use self.RoundClass which points to the correct Round implementation (pokerlib.Round)
            self.round = self.RoundClass(
                round_id, current_players, self.button, self.small_blind, self.big_blind
            )
            # Reset our custom flag (also done in publicOut, but good practice here too)
            self.round_over_flag = False
        except Exception as e:
            print(f"--- ERROR DURING RoundClass INITIALIZATION (Round {round_id}) ---")
            traceback.print_exc()
            print("-----------------------------------------------------")
            self.round = None  # Ensure round is None if init fails

    # --- END OVERRIDE ---

    def _display_game_state(self):
        """ Displays the current state of the table and round. """
        print("-" * 40)
        # Check if round exists and seems valid enough to display
        if ( not self.round
            or not hasattr(self.round, 'id') # Check for basic round attributes
            or not hasattr(self.round, 'players')
            or not isinstance(self.round.players, list) # Ensure players is a list
            or not hasattr(self.round, 'turn')
            or not hasattr(self.round, 'board')
            or not hasattr(self.round, 'pot_size')
            or not isinstance(self.round.pot_size, list)): # Ensure pot_size is list
            print("No active round or essential round state missing.")
            print("Players at table:")
            for player in self.seats.getPlayerGroup():
                print( f"  - {player.name}: ${player.money if hasattr(player, 'money') else 'N/A'}" )
            print("-" * 40)
            return

        try:
            # Access attributes known to exist from checks above
            round_id = self.round.id
            turn_name = self.round.turn.name
            board_cards = format_cards(self.round.board)
            pot_total = sum(self.round.pot_size)
            current_turn_enum = self.round.turn
            players_in_round = self.round.players
            num_players_in_round = len(players_in_round) # Get length for safe indexing

            button_idx = self.round.button if hasattr(self.round, "button") else -1
            sb_idx = ( self.round.small_blind_player_index if hasattr(self.round, "small_blind_player_index") else -1 )
            bb_idx = ( self.round.big_blind_player_index if hasattr(self.round, "big_blind_player_index") else -1 )

        except (AttributeError, TypeError, IndexError) as e:
            # Catch potential errors during attribute access
            print(f"Error accessing round details for display (state changing?): {e}")
            print("Players at table (fallback):")
            for player in self.seats.getPlayerGroup():
                print( f"  - {player.name}: ${player.money if hasattr(player, 'money') else 'N/A'}" )
            print("-" * 40)
            return

        print(f"Round ID: {round_id}, Turn: {turn_name}")
        print(f"Board: [ {board_cards} ]")
        print(f"Total Pot: ${pot_total}")
        print("Players:")


        for idx, player in enumerate(players_in_round):
            # Check player object and essential attributes
            if not player or not all(hasattr(player, attr) for attr in ['id', 'name', 'is_folded', 'is_all_in', 'turn_stake', 'stake', 'money']):
                print(f"  - (Skipping display for invalid player object at index {idx})")
                continue

            status = []
            if player.is_folded: status.append("FOLDED")
            if player.is_all_in: status.append("ALL-IN")

            # Index check before accessing using indices
            player_round_index = idx # Use direct index now

            if player_round_index == button_idx: status.append("DEALER")
            if player_round_index == sb_idx: status.append("SB")
            if player_round_index == bb_idx: status.append("BB")

            turn_stake_val = 0
            # Check turn_stake list validity
            if isinstance(player.turn_stake, list) and len(player.turn_stake) > current_turn_enum.value :
                turn_stake_val = player.turn_stake[current_turn_enum.value]

            stake_info = f"Stake(Turn): ${turn_stake_val}"
            stake_info += f" Stake(Total): ${player.stake}"

            cards_str = "( ? ? )"
            if player.id in self._player_cards:
                cards_str = f"( {format_cards(self._player_cards[player.id])} )"
            elif (self.round_over_flag or self.round.finished) and not player.is_folded and hasattr(player, "cards"): # Check round.finished too
                cards_str = f"( {format_cards(player.cards)} )"

            print( f"  - {player.name}: ${player.money} {cards_str} " f"[{stake_info}] {' '.join(status)}" )
        print("-" * 40)

    # --- Output Handling ---
    def publicOut(self, out_id, **kwargs):
        """Handles public messages broadcast to all players."""
        player_id = kwargs.get("player_id")
        player_name = self._get_player_name(player_id) if player_id else "System"

        self._current_action_player_id = None
        processed = False

        # --- Process RoundPublicOutId First ---
        if isinstance(out_id, RoundPublicOutId):
            processed = True
            if out_id == RoundPublicOutId.NEWROUND:
                print("==> ROUND: Dealing cards...")
            elif out_id == RoundPublicOutId.NEWTURN:
                print(f"\n==> ROUND: --- {kwargs['turn'].name} ---")
                self._display_game_state()
            elif out_id == RoundPublicOutId.SMALLBLIND:
                print(
                    f"==> ROUND: {player_name} posts Small Blind ${kwargs['paid_amount']}."
                )
            elif out_id == RoundPublicOutId.BIGBLIND:
                print(
                    f"==> ROUND: {player_name} posts Big Blind ${kwargs['paid_amount']}."
                )
            elif out_id == RoundPublicOutId.PLAYERCHECK:
                print(f"==> ACTION: {player_name} checks.")
            elif out_id == RoundPublicOutId.PLAYERCALL:
                print(f"==> ACTION: {player_name} calls ${kwargs['paid_amount']}.")
            elif out_id == RoundPublicOutId.PLAYERFOLD:
                print(f"==> ACTION: {player_name} folds.")
            elif out_id == RoundPublicOutId.PLAYERRAISE:
                print(
                    f"==> ACTION: {player_name} raises by ${kwargs['raised_by']} (total bet this turn: ${kwargs['paid_amount']})."
                )
            elif out_id == RoundPublicOutId.PLAYERISALLIN:
                print(f"==> INFO: {player_name} is ALL-IN!")
            elif out_id == RoundPublicOutId.PLAYERWENTALLIN:
                print(
                    f"==> ACTION: {player_name} goes ALL-IN with ${kwargs['paid_amount']}!"
                )
            elif out_id == RoundPublicOutId.PLAYERACTIONREQUIRED:
                print(f"\n>>> ACTION REQUIRED: {player_name}")
                if self.round and hasattr(self.round, "current_player"):
                    cp = None
                    try:
                        cp = self.round.current_player
                    except Exception:
                        pass
                    if cp:
                        print(f"    Amount to call: ${kwargs.get('to_call', 'N/A')}")
                        print(
                            f"    Stack: ${cp.money if hasattr(cp, 'money') else 'N/A'}"
                        )
                    else:
                        print("    (Player info unavailable for action req)")
                else:
                    print("    (Round info unavailable for action req)")
                self._current_action_player_id = player_id
            elif out_id == RoundPublicOutId.PLAYERCHOICEREQUIRED:
                pass  # Ignoring
            elif out_id == RoundPublicOutId.PUBLICCARDSHOW:
                hand_info = ""
                player = (
                    self.round.players.getPlayerById(player_id)
                    if self.round and hasattr(self.round, "players")
                    else None
                )
                if player and hasattr(player, "hand") and player.hand:
                    hand_info = f" ({format_hand_enum(player.hand.handenum)})"
                print(
                    f"==> SHOWDOWN: {player_name} shows {format_cards(kwargs.get('cards',[]))}{hand_info}."
                )
            elif out_id == RoundPublicOutId.DECLAREPREMATUREWINNER:
                print(
                    f"==> WINNER: {player_name} wins ${kwargs['money_won']} as the only remaining player."
                )
            elif out_id == RoundPublicOutId.DECLAREFINISHEDWINNER:
                hand_name = format_hand_enum(kwargs.get("handname"))
                hand_cards = format_cards(kwargs.get("hand", []))
                print(
                    f"==> WINNER: {player_name} wins ${kwargs['money_won']} with {hand_name} ({hand_cards})."
                )
            elif out_id == RoundPublicOutId.ROUNDFINISHED:
                print("\n======= ROUND FINISHED =======")
                self.round_over_flag = True  # SET FLAG
                self._display_game_state()
            elif out_id == RoundPublicOutId.ROUNDCLOSED:
                print("======= ROUND CLOSED (Internal library state) =======")
                self._player_cards.clear()

        # --- Process TablePublicOutId Second ---
        elif isinstance(out_id, TablePublicOutId):
            processed = True
            if out_id == TablePublicOutId.PLAYERJOINED:
                print(f"==> TABLE: {player_name} joined seat {kwargs['player_seat']}.")
            elif out_id == TablePublicOutId.PLAYERREMOVED:
                print(f"==> TABLE: {player_name} left the table.")
            elif out_id == TablePublicOutId.NEWROUNDSTARTED:
                print(f"==> TABLE: Confirmed Start Round (ID: {kwargs['round_id']})")
                # self.round_over_flag = False # RESET FLAG HERE - Now done in _newRound override
            elif out_id == TablePublicOutId.ROUNDNOTINITIALIZED:
                print("==> ERROR: No round is currently running.")
            elif out_id == TablePublicOutId.ROUNDINPROGRESS:
                print("==> ERROR: Round already in progress.")
            elif out_id == TablePublicOutId.INCORRECTNUMBEROFPLAYERS:
                print("==> ERROR: Need at least 2 players to start a round.")

        # --- Catch any unhandled ---
        if not processed:
            if out_id != RoundPublicOutId.PLAYERCHOICEREQUIRED:
                print(
                    f"Unhandled Public Out: ID={out_id} (Type: {type(out_id).__name__}) Data: {kwargs}"
                )

    def privateOut(self, player_id, out_id, **kwargs):
        """Handles private messages sent only to a specific player."""
        player_name = self._get_player_name(player_id)

        if out_id == RoundPrivateOutId.DEALTCARDS:
            cards = kwargs.get("cards")
            if cards:
                self._player_cards[player_id] = cards
                print(
                    f"==> PRIVATE [{player_name}]: You are dealt {format_cards(cards)}"
                )
            else:
                print(
                    f"==> PRIVATE [{player_name}]: Dealing cards error (no cards data)."
                )
        elif out_id == TablePrivateOutId.BUYINTOOLOW:
            print(f"==> ERROR [{player_name}]: Buy-in of ${self.buyin} required.")
        elif out_id == TablePrivateOutId.TABLEFULL:
            print(f"==> ERROR [{player_name}]: Table is full.")
        elif out_id == TablePrivateOutId.PLAYERALREADYATTABLE:
            print(f"==> ERROR [{player_name}]: You are already at the table.")
        elif out_id == TablePrivateOutId.PLAYERNOTATTABLE:
            print(f"==> ERROR [{player_name}]: You are not at this table.")
        elif out_id == TablePrivateOutId.INCORRECTSEATINDEX:
            print(f"==> ERROR [{player_name}]: Invalid seat index or seat is taken.")


# --- Main Game Logic ---


def get_player_action(player_name, to_call, player_money, can_check, can_raise):
    """Prompts the user for an action and validates it."""
    print(f"--- {player_name}'s Turn ---")
    actions = ["FOLD"]
    action_prompt = "FOLD"

    if can_check:
        actions.append("CHECK")
        action_prompt += "/CHECK"
    elif to_call > 0 and player_money > 0:
        actions.append("CALL")
        effective_call = min(to_call, player_money)
        action_prompt += f"/CALL({effective_call})"

    if can_raise:
        actions.append("RAISE")
        action_prompt += "/RAISE"

    print(f"Available actions: {action_prompt}")

    while True:
        action_str = input("Enter action: ").upper().strip()

        if action_str == "CALL" and can_check:
            print("No bet to call. Use CHECK.")
            continue
        if action_str == "CHECK" and not can_check:
            print(f"Cannot check. Bet is ${to_call}. Use CALL or FOLD.")
            continue
        if action_str not in actions:
            print("Invalid action. Choose from:", ", ".join(actions))
            continue

        if action_str == "RAISE":
            if not can_raise:
                print("Error: Raise action selected but not available.")
                continue

            min_raise = table.big_blind
            max_raise = player_money - to_call

            if max_raise < min_raise and player_money > to_call:
                min_raise = max_raise

            if max_raise <= 0:
                print("Cannot raise, not enough funds after calling.")
                continue

            while True:
                try:
                    prompt_range = (
                        f"(min {min_raise}, max {max_raise})"
                        if min_raise < max_raise
                        else f"(exactly {max_raise} to go all-in)"
                    )
                    raise_by_str = input(f"Raise BY how much? {prompt_range}: ")
                    raise_by = int(raise_by_str)

                    is_all_in_raise = to_call + raise_by >= player_money

                    if raise_by <= 0:
                        print("Raise amount must be positive.")
                    elif raise_by > max_raise:
                        print(
                            f"You only have ${max_raise} available after calling. Max raise BY is {max_raise}."
                        )
                    elif raise_by < min_raise and not is_all_in_raise:
                        print(
                            f"Minimum raise BY is ${min_raise} (unless going all-in)."
                        )
                    else:
                        actual_raise_by = min(raise_by, max_raise)
                        return RoundPublicInId.RAISE, {"raise_by": actual_raise_by}
                except ValueError:
                    print("Invalid amount. Please enter a number.")
        elif action_str == "FOLD":
            return RoundPublicInId.FOLD, {}
        elif action_str == "CHECK":
            if can_check:
                return RoundPublicInId.CHECK, {}
        elif action_str == "CALL":
            if not can_check:
                return RoundPublicInId.CALL, {}

        print("Error processing action. Please try again.")


# --- Main Execution ---

if __name__ == "__main__":
    NUM_PLAYERS = 3
    BUYIN = 200
    SMALL_BLIND = 5
    BIG_BLIND = 10

    # 1. Create the Table - Use _id
    table = CommandLineTable(
        _id=0,  # Use _id parameter for Table constructor
        seats=PlayerSeats([None] * NUM_PLAYERS),
        buyin=BUYIN,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND,
    )

    # 2. Create Players
    players = []
    for i in range(NUM_PLAYERS):
        player_name = f"Player_{i+1}"
        player = Player(
            table_id=table.id,  # Player uses table_id which references Table's _id
            _id=i + 1,
            name=player_name,
            money=BUYIN,
        )
        players.append(player)
        table.publicIn(player.id, TablePublicInId.BUYIN, player=player)

    print("\n--- Welcome to Simple Pokerlib Game! ---")
    print(f"{NUM_PLAYERS} players joined with ${BUYIN} each.")
    print(f"Blinds: ${SMALL_BLIND}/${BIG_BLIND}")

    # 3. Game Loop
    round_count = 0
    try:
        while True:
            active_players_obj = table.seats.getPlayerGroup()
            if len(active_players_obj) < 2:
                print("\nNot enough players to continue. Game over.")
                break

            round_count += 1
            print(f"\n--- Starting Round {round_count} ---")

            initiator = active_players_obj[0] if active_players_obj else None
            if not initiator:
                print("Error: No players left to initiate round.")
                break
            # Start the round using the table's publicIn
            table.publicIn(
                initiator.id, TablePublicInId.STARTROUND, round_id=round_count
            )

            # Inner loop for actions within a round - Uses custom flag
            while table.round and not table.round_over_flag:
                action_player_id_to_process = table._current_action_player_id

                if action_player_id_to_process:
                    table._current_action_player_id = None  # Clear flag early

                    player = table.seats.getPlayerById(action_player_id_to_process)
                    if not player:
                        print(
                            f"Warning: Action requested for missing player ID {action_player_id_to_process}"
                        )
                        continue

                    current_player_obj = None
                    if table.round and hasattr(table.round, "current_player"):
                        try:
                            current_player_obj = table.round.current_player
                        except Exception:
                            pass

                    if current_player_obj and player.id == current_player_obj.id:
                        # Check attributes before calling get_player_action
                        if (
                            not all(
                                hasattr(player, attr) for attr in ["money", "stake"]
                            )
                            or not (
                                hasattr(player, "turn_stake")
                                and isinstance(player.turn_stake, list)
                            )
                            or not (table.round and hasattr(table.round, "to_call"))
                        ):
                            print(
                                f"Warning: Player {player.name} or round missing critical attributes."
                            )
                            time.sleep(0.1)  # Wait briefly in case state is updating
                            table._current_action_player_id = action_player_id_to_process  # Reset flag to retry? Or just continue? Let's continue for now.
                            continue

                        to_call = table.round.to_call
                        can_check = to_call == 0
                        can_raise = player.money > to_call

                        action_enum, action_kwargs = get_player_action(
                            player.name, to_call, player.money, can_check, can_raise
                        )
                        # Send action back to table
                        table.publicIn(player.id, action_enum, **action_kwargs)

                    elif (
                        current_player_obj
                    ):  # Log only if mismatch detected and current exists
                        req_for = f"{player.name} ({action_player_id_to_process})"
                        curr = f"{current_player_obj.name} ({current_player_obj.id})"
                        print(
                            f"Warning: Action request mismatch. Requested for {req_for}, current is {curr}"
                        )

                time.sleep(0.02)

            # <<< Code after inner while loop (round ended based on flag) >>>
            # Clear the library's round reference AFTER our loop using the flag finishes
            if table.round:
                # print("[DEBUG] Manually clearing table.round reference.") # Optional
                table.round = None

            print("\nRound ended state. Current stacks:")
            final_players = table.seats.getPlayerGroup()
            if not final_players:
                print("  No players remaining at the table.")
            else:
                for p in final_players:
                    money_val = p.money if hasattr(p, "money") else "N/A"
                    print(f"  - {p.name}: ${money_val}")

            # Ask to continue
            try:
                cont = input("\nPlay another round? (y/n): ").lower()
                if cont != "y":
                    break
            except EOFError:
                print("\nInput closed unexpectedly. Exiting.")
                break

    except Exception as e:
        print("\n--- UNEXPECTED ERROR OCCURRED ---")
        traceback.print_exc()
        print("---------------------------------")

    finally:
        print("\n--- Game Ended ---")
        print("Final Stacks:")
        final_players = table.seats.getPlayerGroup()
        if not final_players:
            print("  No players remaining.")
        else:
            for p in final_players:
                money_str = f"${p.money}" if hasattr(p, "money") else "N/A"
                print(f"  - {p.name}: {money_str}")
