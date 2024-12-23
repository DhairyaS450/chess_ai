import random
import logging

# Configure logging
logging.basicConfig(
    filename='chess_ai_logs.txt',  # Log file name
    level=logging.DEBUG,          # Log all DEBUG level and above messages
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)


class ChessAI:
    def __init__(self, difficulty='easy'):
        """
        Initializes the ChessAI with a specified difficulty level.
        Args:
            difficulty (str): 'easy', 'medium', or 'hard'.
        """
        self.difficulty = difficulty

    def choose_move(self, board):
        """
        Chooses a move for the AI based on the difficulty level.
        Args:
            board (ChessBoard): The current game board.
        Returns:
            tuple: The chosen move ((start_row, start_col), (end_row, end_col)).
        """
        if self.difficulty == 'easy':
            return self._random_move(board)
        elif self.difficulty == 'medium':
            return self._basic_evaluation(board)
        elif self.difficulty == 'hard':
            pieces_count = sum(len(positions) for positions in board.piece_positions.values())
            depth = 3 if pieces_count > 20 else 4  # Early game = depth 3, Endgame = depth 4+
            _, best_move = self._minimax(board, depth=depth)
            return best_move
        else:
            raise ValueError("Invalid difficulty level.")

    import random

    def _random_move(self, board):
        """
        Chooses a random legal move for the AI.
        Args:
            board (ChessBoard): The current game board.
        Returns:
            tuple: A randomly chosen legal move.
        """
        board.display_board()

        legal_moves = board.generate_legal_moves(board.turn)
        return random.choice(legal_moves) if legal_moves else None

    def _basic_evaluation(self, board):
        """
        Chooses a move based on a basic evaluation function.
        Args:
            board (ChessBoard): The current game board.
        Returns:
            tuple: The move with the highest evaluation score.
        """
        def evaluate_move(move):
            start, end = move
            target_piece = board.board[end[0]][end[1]]
            # Assign values to pieces (higher for more valuable pieces)
            piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
            return piece_values.get(target_piece[1], 0) if target_piece != '..' else 0

        legal_moves = board.generate_legal_moves(board.turn)
        if not legal_moves:
            return None

        # Choose the move with the highest evaluation
        return max(legal_moves, key=evaluate_move)

    def _minimax(self, board, depth, maximizing_player=True, alpha=float('-inf'), beta=float('inf')):
        """
        Implements the minimax algorithm with alpha-beta pruning.
        Args:
            board (ChessBoard): The current game board.
            depth (int): The search depth.
            maximizing_player (bool): True if the AI is maximizing, False if minimizing.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
        Returns:
            tuple: (evaluation, move) - The best evaluation and the corresponding move.
        """
        if depth == 0 or board.is_checkmate(board.turn) or board.is_stalemate(board.turn):
            return self._evaluate_board(board), None
    
        legal_moves = board.generate_legal_moves(board.turn)
        if not legal_moves:
            return self._evaluate_board(board), None
            
        # logging.debug(f"Depth: {depth}, Maximizing: {maximizing_player}, Legal Moves: {legal_moves}")
        # logging.debug(f"{board.board}")
    
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in legal_moves:
                board.execute_move(move)  # Apply the move
                eval, _ = self._minimax(board, depth - 1, False, alpha, beta)
                board.undo_move()  # Undo the move
    
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:  # Alpha-beta pruning
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in legal_moves:
                board.execute_move(move)  # Apply the move
                eval, _ = self._minimax(board, depth - 1, True, alpha, beta)
                board.undo_move()  # Undo the move
    
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:  # Alpha-beta pruning
                    break
            return min_eval, best_move

    def _evaluate_board(self, board):
        """
        Evaluates the board position for the AI.
        Args:
            board (ChessBoard): The current game board.
        Returns:
            int: The evaluation score (positive for advantage, negative for disadvantage).
        """
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        
        pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 5, 5, 5, 5, 5, 5, 5,
            1, 1, 2, 3, 3, 2, 1, 1,
            0, 0, 0, 2, 2, 0, 0, 0,
            0, 0, 0, 1, 1, 0, 0, 0,
            0, 0.25, 0, -1, -1, 0, 0.25, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ]
        
        knight_table = [
            -5, -4, -3, -3, -3, -3, -4, -5,
            -4, -2,  0,  0,  0,  0, -2, -4,
            -3,  0,  1,  1.5, 1.5,  1,  0, -3,
            -3,  0.5, 1.5, 2, 2, 1.5, 0.5, -3,
            -3,  0,  1.5, 2, 2, 1.5,  0, -3,
            -3,  0.5, 1.25, 1, 1,  1.25, 0.5, -3,
            -4, -2,  0, 0.5, 0.5,  0, -2, -4,
            -5, -4, -3, -3, -3, -3, -4, -5,
        ]
        
        bishop_table = [
            -2, -1, -1, -1, -1, -1, -1, -2,
            -1,  0,  0,  0,  0,  0,  0, -1,
            -1,  0,  0.5,  1,  1, 0.5,  0, -1,
            -1,  0.5, 0.5,  1,  1, 0.5, 0.5, -1,
            -1,  0,  1,  1,  1,  1,  0, -1,
            -1,  1,  1,  1,  1,  1,  1, -1,
            -1,  1,  0,  0,  0,  0,  1, -1,
            -2, -1, -1, -1, -1, -1, -1, -2,
        ]

        rook_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0.5, 1, 1, 1, 1, 1, 1, 0.5,
            -0.5, 0, 0, 0, 0, 0, 0, -0.5,
            -0.5, 0, 0, 0, 0, 0, 0, -0.5,
            -0.5, 0, 0, 0, 0, 0, 0, -0.5,
            -0.5, 0, 0, 0, 0, 0, 0, -0.5,
            -0.5, 0, 0, 0, 0, 0, 0, -0.5,
            0, 0, 0, 0.5, 0.5, 0, 0, 0,
        ]

        queen_table = [
            -2, -1, -1, -0.5, -0.5, -1, -1, -2,
            -1,  0,  0,  0,  0,  0,  0, -1,
            -1,  0,  0.5,  0.5,  0.5,  0.5,  0, -1,
            -0.5, 0,  0.5,  0.5,  0.5,  0.5,  0, -0.5,
            0,  0,  0.5,  0.5,  0.5,  0.5,  0, -0.5,
            -1,  0.5,  0.5,  0.5,  0.5,  0.5,  0, -1,
            -1,  0,  0.5,  0,  0,  0,  0, -1,
            -2, -1, -1, -0.5, -0.5, -1, -1, -2,
        ]

        king_early_table = [
            -3, -4, -4, -5, -5, -4, -4, -3,
            -3, -4, -4, -5, -5, -4, -4, -3,
            -3, -4, -4, -5, -5, -4, -4, -3,
            -3, -4, -4, -5, -5, -4, -4, -3,
            -2, -3, -3, -4, -4, -3, -3, -2,
            -1, -2, -2, -2, -2, -2, -2, -1,
            2,  2,  0,  0,  0,  0,  2,  2,
            2,  3,  1,  0,  0,  1,  3,  2,
        ]

        king_endgame_table = [
            -5, -4, -3, -2, -2, -3, -4, -5,
            -3, -2, -1,  0,  0, -1, -2, -3,
            -2, -1,  1,  2,  2,  1, -1, -2,
            -1,  0,  2,  3,  3,  2,  0, -1,
            0,  1,  2,  3,  3,  2,  1,  0,
            1,  2,  3,  4,  4,  3,  2,  1,
            2,  3,  4,  5,  5,  4,  3,  2,
            3,  4,  5,  5,  5,  5,  4,  3,
        ]
        score = 0
        
        for piece, positions in board.piece_positions.items():
            piece_value = piece_values.get(piece[1], 0)
            for pos in positions:
                row, col = pos
                table_value = 0
    
                # Use positional value if applicable
                if piece[1] == 'P':  # Pawn
                    if piece[0] == 'w':  # White pawn
                        table_value = pawn_table[row * 8 + col]
                    else:  # Black pawn
                        table_value = pawn_table[(7 - row) * 8 + col]  # Flip rows for black
    
                # Add table value and piece value to score
                if piece[0] == 'w':  # White pieces
                    score -= piece_value + table_value
                else:  # Black pieces
                    score += piece_value + table_value

        return score