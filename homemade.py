"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.types import MOVE, HOMEMADE_ARGS_TYPE
import logging
import numpy as np
import torch
import os
from EvaluationNetwork import EvaluationNetwork

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

class RedChessAI(MinimalEngine):

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return '0000'
        
        move = random.choice(legal_moves)
        return PlayResult(move, None)
    
# Function to convert the chess board state to a tensor representation
def board_to_tensor(board):
    # Initialize a tensor of shape (14, 8, 8) filled with zeros
    board_tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Mapping from piece type to index in the tensor channels
    piece_indices = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # Map pieces on the board to the tensor
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Determine the offset for the piece color
            color_offset = 0 if piece.color == chess.WHITE else 6
            # Get the index for the piece type
            piece_index = piece_indices[piece.piece_type] + color_offset
            # Convert the square index to 2D coordinates (x, y)
            x, y = divmod(square, 8)
            # Set the presence of the piece in the tensor
            board_tensor[piece_index, x, y] = 1.0
    
    # Initialize attack maps for white and black pieces
    white_attacks = np.zeros((8, 8), dtype=np.float32)
    black_attacks = np.zeros((8, 8), dtype=np.float32)
    
    # Generate attack maps
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Get the squares attacked by the piece
            attacks = board.attacks(square)
            for attacked_square in attacks:
                x, y = divmod(attacked_square, 8)
                if piece.color == chess.WHITE:
                    white_attacks[x, y] = 1.0
                else:
                    black_attacks[x, y] = 1.0
    
    # Add attack maps to the tensor
    board_tensor[12] = white_attacks
    board_tensor[13] = black_attacks
    
    # Convert the numpy array to a PyTorch tensor and add batch dimension
    return torch.from_numpy(board_tensor).unsqueeze(0).to(device)


# These should be defined inside of RCAI_RL1, but MinimalEngine requires a
# configuration file I've yet to familiarize myself with
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.path.join("MODELS", "RedChessAI20241005164331.pth")
model = EvaluationNetwork().to(device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

class RCAI_RL1(MinimalEngine):
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE):
        legal_moves = list(board.legal_moves)
        move_values = []

        # Evaluate all legal moves using current_model
        for move in legal_moves:
            board.push(move)
            # TODO: Minimax algorithm somewhere in the next 3 lines of code
            # Makes the board machine readable
            state_tensor = board_to_tensor(board)
            # How strong the AI thinks this board position is
            value = model(state_tensor).item()
            # Storing the value of the board for later analysis
            move_values.append(value) 
            board.pop()

        # Apply softmax to the predicted values
        logits = np.array(move_values)
        probabilities = np.exp(2*logits)
        probabilities /= probabilities.sum()

        # Select a move based on the probabilities
        move_index = np.random.choice(len(legal_moves), p=probabilities)
        chosen_move = legal_moves[move_index]
        return PlayResult(chosen_move, None)

# Below are the example engines 

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
    pass

# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)
