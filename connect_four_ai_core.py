import numpy as np
import random
import math
import streamlit as st # Streamlit is imported here but logic is used in app.py

# --- Game Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
WINDOW_LENGTH = 4 

# --- Board Helper Functions ---

def is_valid_location(board, col):
    """Checks if the top row of a column is empty."""
    return board[ROW_COUNT - 1][col] == EMPTY

def get_next_open_row(board, col):
    """Finds the lowest empty row in the given column."""
    for r in range(ROW_COUNT):
        if board[r][col] == EMPTY:
            return r

def get_valid_locations(board):
    """Returns a list of columns where a piece can be dropped."""
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def is_winning_move(board, piece):
    """Checks for 4-in-a-row (horizontal, vertical, and both diagonals)."""
    # Check Horizontal locations
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if all(board[r][c + i] == piece for i in range(4)): return True
    # Check Vertical locations
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board[r + i][c] == piece for i in range(4)): return True
    # Check Positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all(board[r + i][c + i] == piece for i in range(4)): return True
    # Check Negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all(board[r - i][c + i] == piece for i in range(4)): return True
    return False

def is_terminal_node(board):
    """Checks if the game is over (win or draw)."""
    return is_winning_move(board, PLAYER_PIECE) or is_winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

# --- Heuristic Function ---

def evaluate_window(window, piece):
    """
    The core heuristic evaluation: assigns a score to a 4-piece window.
    This is the *intelligence* of your AI.
    """
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100000 
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 50 
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 5

    # Strongly penalize immediate wins for the opponent
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4000 
        
    return score

def score_position(board, piece):
    """
    Calculates the total score for the entire board by checking all possible 4-piece windows.
    """
    score = 0

    # 1. Score Center Column (Good control heuristic)
    center_array = list(board[:, COLUMN_COUNT // 2])
    center_count = center_array.count(piece)
    score += center_count * 3

    # 2. Score Horizontal
    for r in range(ROW_COUNT):
        row_array = list(board[r, :])
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # 3. Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = list(board[:, c])
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # 4. Score Diagonals (Positive and Negative slopes)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window_pos = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window_pos, piece)
            
            window_neg = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window_neg, piece)

    return score


# --- Minimax Algorithm ---

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    The recursive adversarial search algorithm with Alpha-Beta Pruning.
    """
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if is_winning_move(board, AI_PIECE):
                # AI wins (return a very large score)
                return (None, 100000000000000)
            elif is_winning_move(board, PLAYER_PIECE):
                # Player wins (return a very large negative score)
                return (None, -100000000000000)
            else: 
                # Draw or Board Full
                return (None, 0)
        else: 
            # Depth limit reached: use the heuristic
            return (None, score_position(board, AI_PIECE))

    # Maximizing Player (AI)
    if maximizing_player:
        value = -math.inf
        column = random.choice(valid_locations) 

        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            temp_board[row][col] = AI_PIECE
            
            # Recursively call minimax for the opponent (False)
            new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
            
            if new_score > value:
                value = new_score
                column = col
            
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Alpha-Beta Pruning
                
        return column, value

    # Minimizing Player (Human)
    else: 
        value = math.inf
        column = random.choice(valid_locations)
        
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            temp_board[row][col] = PLAYER_PIECE

            # Recursively call minimax for the AI (True)
            new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]

            if new_score < value:
                value = new_score
                column = col
                
            beta = min(beta, value)
            if alpha >= beta:
                break # Alpha-Beta Pruning
                
        return column, value