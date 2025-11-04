import numpy as np
import streamlit as st
import math
from connect_four_ai_core import (
    minimax, is_valid_location, get_next_open_row, get_valid_locations, 
    is_winning_move, AI_PIECE, PLAYER_PIECE, EMPTY, ROW_COUNT, COLUMN_COUNT
)

# --- Streamlit Application ---

# Define the search depth here. This is a great tunable parameter for your presentation!
SEARCH_DEPTH = 4 

def draw_board(board):
    """Draws the Connect Four board using Streamlit's layout features."""
    
    st.markdown(f"## Connect Four Minimax AI (Depth {SEARCH_DEPTH})")
    
    # Check for game over state and display result
    if 'game_over' in st.session_state and st.session_state.game_over:
        if st.session_state.winner == PLAYER_PIECE:
            st.success("🎉 You Win! (You beat the Minimax algorithm!)")
        elif st.session_state.winner == AI_PIECE:
            st.error("🤖 The AI Wins! (Minimax prevails)")
        else:
            st.warning("It's a Draw!")
        
        # Reset button
        if st.button('Play Again', key='reset_button'):
            # Reset all session state variables
            st.session_state.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
            st.session_state.game_over = False
            st.session_state.turn = PLAYER_PIECE
            st.session_state.winner = EMPTY
            st.rerun()
        return

    # Game Info Display
    turn_text = "Your Turn (Blue)" if st.session_state.turn == PLAYER_PIECE else "AI's Turn (Red)"
    st.info(f"Current Turn: **{turn_text}**")
    
    # Create the board grid using columns
    cols = st.columns(COLUMN_COUNT)
    
    for c in range(COLUMN_COUNT):
        with cols[c]:
            # Player move logic and button
            def player_move_callback(col=c):
                if st.session_state.turn == PLAYER_PIECE and is_valid_location(st.session_state.board, col):
                    row = get_next_open_row(st.session_state.board, col)
                    st.session_state.board[row][col] = PLAYER_PIECE
                    st.session_state.turn = AI_PIECE # Switch turn
                    
            disabled = not is_valid_location(board, c) or st.session_state.turn == AI_PIECE # Disable if full or if it's AI's turn
            
            st.button(f"Column {c+1}", key=f"col_{c}", on_click=player_move_callback, disabled=disabled)

            # Display the board pieces (from top to bottom)
            for r in range(ROW_COUNT - 1, -1, -1):
                piece = board[r][c]
                if piece == PLAYER_PIECE:
                    # Blue circle for Human player
                    st.markdown('<p style="font-size: 40px; text-align: center; color: blue;">●</p>', unsafe_allow_html=True)
                elif piece == AI_PIECE:
                    # Red circle for AI player
                    st.markdown('<p style="font-size: 40px; text-align: center; color: red;">●</p>', unsafe_allow_html=True)
                else:
                    # Light gray circle for empty spot
                    st.markdown('<p style="font-size: 40px; text-align: center; color: #f0f0f0;">●</p>', unsafe_allow_html=True)

def check_game_state(board):
    """Check for win conditions after a move."""
    if is_winning_move(board, PLAYER_PIECE):
        st.session_state.game_over = True
        st.session_state.winner = PLAYER_PIECE
    elif is_winning_move(board, AI_PIECE):
        st.session_state.game_over = True
        st.session_state.winner = AI_PIECE
    elif len(get_valid_locations(board)) == 0:
        st.session_state.game_over = True
        st.session_state.winner = EMPTY # Draw

def ai_turn_logic():
    """Executes the AI's Minimax move."""
    
    with st.spinner(f"🤖 AI is evaluating {2 * SEARCH_DEPTH} half-moves..."):
        # Call the Minimax search function from the core logic file
        col, minimax_score = minimax(st.session_state.board, SEARCH_DEPTH, -math.inf, math.inf, True) 

    if col is not None and is_valid_location(st.session_state.board, col):
        row = get_next_open_row(st.session_state.board, col)
        st.session_state.board[row][col] = AI_PIECE
        st.session_state.turn = PLAYER_PIECE # Switch back to human
        st.rerun() 
    else:
        # If no valid move is found (should be a draw)
        check_game_state(st.session_state.board)


def main():
    """Initializes and runs the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Connect Four Minimax AI")

    # --- Initialize Session State (Critical for Streamlit) ---
    if 'board' not in st.session_state:
        st.session_state.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'turn' not in st.session_state:
        st.session_state.turn = PLAYER_PIECE # Human starts
    if 'winner' not in st.session_state:
        st.session_state.winner = EMPTY
        
    # --- Game Flow ---
    draw_board(st.session_state.board)
    check_game_state(st.session_state.board)
    
    # AI Logic: Run AI turn if it's the AI's turn and the game isn't over
    if st.session_state.turn == AI_PIECE and not st.session_state.game_over:
        ai_turn_logic()
    
    st.markdown("---")
    st.caption("Heuristic functions are what enable the AI to make intelligent decisions without searching to the end of the game.")

if __name__ == "__main__":
    main()