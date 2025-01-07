import random
import logging
import time

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
        self.transposition_table = {}

    def _store_in_transposition_table(self, board_hash, depth, score, move):
        """
        Stores a board evaluation in the transposition table.
        """
        self.transposition_table[board_hash] = {
            'depth': depth,
            'score': score,
            'move': move,
        }

    def _retrieve_from_transposition_table(self, board_hash, depth):
        """
        Retrieves an entry from the transposition table if available.
        """
        entry = self.transposition_table.get(board_hash)
        if entry and entry['depth'] >= depth:
            return entry
        return None

    def choose_move(self, board):
        """
        Chooses a move for the AI based on the difficulty level.
        Args:
            board (ChessBoard): The current game board.
        Returns:
            tuple: The chosen move ((start_row, start_col), (end_row, end_col)).
        """
        if self.difficulty == 'easy':
            return self._basic_evaluation(board)
        elif self.difficulty == 'medium':
            return self._minimax(board, depth=3)
        elif self.difficulty == 'hard':
            best_move = self._iterative_deepening(board, max_depth=10)
            return best_move
        else:
            raise ValueError("Invalid difficulty level.")

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
    
    def _iterative_deepening(self, board, max_depth, time_limit=5):
        """
        Performs iterative deepening search up to a maximum depth.
        Args:
            board (ChessBoard): The current game board.
            max_depth (int): The maximum depth to search.
        Returns:
            tuple: The best move found ((start_row, start_col), (end_row, end_col)).
        """
        best_move = None
        start_time = time.time()
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
            _, best_move = self._minimax(board, depth)
            logging.debug(f"Depth {depth} completed. Best move: {best_move}")
        print(best_move)
        return best_move

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
        board_hash = board.current_hash
        transposition_entry = self._retrieve_from_transposition_table(board_hash, depth)
        if transposition_entry:
            print('Transposition table hit!')
            return transposition_entry['score'], transposition_entry['move']

        if depth == 0 or board.is_checkmate(board.turn) or board.is_stalemate(board.turn):
            score = self._evaluate_board(board)
            self._store_in_transposition_table(board_hash, depth, score, None)
            return score, None

        legal_moves = self._order_moves(board, board.generate_legal_moves(board.turn))
        if not legal_moves:
            score = self._evaluate_board(board)
            self._store_in_transposition_table(board_hash, depth, score, None)
            return score, None
            
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
            self._store_in_transposition_table(board_hash, depth, max_eval, best_move)
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
            self._store_in_transposition_table(board_hash, depth, min_eval, best_move)
            return min_eval, best_move
        
    def _order_moves(self, board, legal_moves):
        """
        Orders moves to improve alpha-beta pruning efficiency by prioritizing captures, checks, promotions, and threats.
        Args:
            board (ChessBoard): The current game board.
            legal_moves (list): List of legal moves.
        Returns:
            list: Ordered list of moves
        """    
        def move_score(move):
            start, end = move
            moving_piece = board.board[start[0]][start[1]]
            captured_piece = board.board[end[0]][end[1]]

            # Assign values to pieces for evaluation
            piece_values = {'P': 1, 'N': 3, 'B': 3.5, 'R': 5, 'Q': 9, 'K': 0}

            score = 0

            # Prioritize captures
            if captured_piece != '..':
                score += 10 * piece_values.get(captured_piece[1], 0) - piece_values.get(moving_piece[1], 0)

            # Prioritize checks
            if board.is_check(move):
                score += 10

            # Prioritize promotions
            if moving_piece[1] == 'P' and (end[0] == 0 or end[0] == 7):
                score += 25

            # Threatening moves (attacking high-value pieces)
            for target in board.piece_positions.get('b' if moving_piece[0] == 'w' else 'w', []):
                if target == end:
                    target_piece = board.board[target[0]][target[1]]
                    score += 5 * piece_values.get(target_piece[1], 0)

            # Positional advantages (e.g., center control)
            center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
            if end in center_squares:
                score += 3

            return score
        
        return sorted(legal_moves, key=move_score, reverse=True)

    def _endgame_positional_value(self, piece, row, col):
        """
        Get the endgame positional value of a piece based on its location on the board
        Args:
            piece (str): The chess piece (eg. wB, bN)
            row (int): Row of the piece on the board.
            col (int): Column of the piece on the board
        Returns:
            int: Table value of piece
        """
        eg_pawn_table = [
            0,   0,   0,   0,   0,   0,   0,   0,
            178, 173, 158, 134, 147, 132, 165, 187,
            94, 100,  85,  67,  56,  53,  82,  84,
            32,  24,  13,   5,  -2,   4,  17,  17,
            13,   9,  -3,  -7,  -7,  -8,   3,  -1,
            4,   7,  -6,   1,   0,  -5,  -1,  -8,
            13,   8,   8,  10,  13,   0,   2,  -7,
            0,   0,   0,   0,   0,   0,   0,   0,
        ]

        eg_knight_table = [
            -58, -38, -13, -28, -31, -27, -63, -99,
            -25,  -8, -25,  -2,  -9, -25, -24, -52,
            -24, -20,  10,   9,  -1,  -9, -19, -41,
            -17,   3,  22,  22,  22,  11,   8, -18,
            -18,  -6,  16,  25,  16,  17,   4, -18,
            -23,  -3,  -1,  15,  10,  -3, -20, -22,
            -42, -20, -10,  -5,  -2, -20, -23, -44,
            -29, -51, -23, -15, -22, -18, -50, -64,
        ]

        eg_bishop_table = [
            -14, -21, -11,  -8, -7,  -9, -17, -24,
            -8,  -4,   7, -12, -3, -13,  -4, -14,
            2,  -8,   0,  -1, -2,   6,   0,   4,
            -3,   9,  12,   9, 14,  10,   3,   2,
            -6,   3,  13,  19,  7,  10,  -3,  -9,
            -12,  -3,   8,  10, 13,   3,  -7, -15,
            -14, -18,  -7,  -1,  4,  -9, -15, -27,
            -23,  -9, -23,  -5, -9, -16,  -5, -17,
        ]

        eg_rook_table = [
            13, 10, 18, 15, 12,  12,   8,   5,
            11, 13, 13, 11, -3,   3,   8,   3,
            7,  7,  7,  5,  4,  -3,  -5,  -3,
            4,  3, 13,  1,  2,   1,  -1,   2,
            3,  5,  8,  4, -5,  -6,  -8, -11,
            -4,  0, -5, -1, -7, -12,  -8, -16,
            -6, -6,  0,  2, -9,  -9, -11,  -3,
            -9,  2,  3, -1, -5, -13,   4, -20,
        ]

        eg_queen_table = [
            -9,  22,  22,  27,  27,  19,  10,  20,
            -17,  20,  32,  41,  58,  25,  30,   0,
            -20,   6,   9,  49,  47,  35,  19,   9,
            3,  22,  24,  45,  57,  40,  57,  36,
            -18,  28,  19,  47,  31,  34,  39,  23,
            -16, -27,  15,   6,   9,  17,  10,   5,
            -22, -23, -30, -16, -16, -23, -36, -32,
            -33, -28, -22, -43,  -5, -32, -20, -41,
        ]

        eg_king_table = [
            -74, -35, -18, -18, -11,  15,   4, -17,
            -12,  17,  14,  17,  17,  38,  23,  11,
            10,  17,  23,  15,  20,  45,  44,  13,
            -8,  22,  24,  27,  26,  33,  26,   3,
            -18,  -4,  21,  24,  27,  23,   9, -11,
            -19,  -3,  11,  21,  23,  16,   7,  -9,
            -27, -11,   4,  13,  14,   4,  -5, -17,
            -53, -34, -21, -11, -28, -14, -24, -43
        ] 

        if piece[1] == 'P':  # Pawn
            if piece[0] == 'w':  # White pawn
                table_value = eg_pawn_table[row * 8 + col]
            else:  # Black pawn
                table_value = eg_pawn_table[(7 - row) * 8 + col]  # Flip rows for black

        elif piece[1] == 'B':  # Bishop
            if piece[0] == 'w': # White bishop
                table_value = eg_bishop_table[row * 8 + col]
            else:
                table_value = eg_bishop_table[(7 - row) * 8 + col]  # Flip rows for black
        
        elif piece[1] == 'N':  # Knight
            if piece[0] == 'w': # White knight
                table_value = eg_knight_table[row * 8 + col]
            else:
                table_value = eg_knight_table[(7 - row) * 8 + col]  # Flip rows for black

        elif piece[1] == 'K':  # King
            if piece[0] == 'w': # White king
                table_value = eg_king_table[row * 8 + col]
            else:
                table_value = eg_king_table[(7 - row) * 8 + col]  # Flip rows for black

        elif piece[1] == 'R': # Rook
            if piece[0] == 'w': # White rook
                table_value = eg_rook_table[row * 8 + col]
            else:
                table_value = eg_rook_table[(7 - row) * 8 + col]  # Flip rows for black
        
        elif piece[1] == 'Q': # Queen
            if piece[0] == 'w': # White queen
                table_value = eg_queen_table[row * 8 + col]
            else:
                table_value = eg_queen_table[(7 - row) * 8 + col]  # Flip rows for black

        return table_value

    def _middlegame_positional_value(self, piece, row, col):
        """
        Get the middlegame positional value of a piece based on its location on the board
        Args:
            piece (str): The chess piece (eg. wB, bN)
            row (int): Row of the piece on the board.
            col (int): Column of the piece on the board
        Returns:
            int: Table value of piece
        """
        mg_pawn_table = [
            0,   0,   0,   0,   0,   0,  0,   0,
            98, 134,  61,  95,  68, 126, 34, -11,
            -6,   7,  26,  31,  65,  56, 25, -20,
            -14,  13,   6,  21,  23,  12, 17, -23,
            -27,  -2,  -5,  12,  17,   6, 10, -25,
            -26,  -4,  -4, -10,   3,   3, 33, -12,
            -35,  -1, -20, -23, -15,  24, 38, -22,
            0,   0,   0,   0,   0,   0,  0,   0,
        ]

        mg_knight_table = [
            -167, -89, -34, -49,  61, -97, -15, -107,
            -73, -41,  72,  36,  23,  62,   7,  -17,
            -47,  60,  37,  65,  84, 129,  73,   44,
            -9,  17,  19,  53,  37,  69,  18,   22,
            -13,   4,  16,  13,  28,  19,  21,   -8,
            -23,  -9,  12,  10,  19,  17,  25,  -16,
            -29, -53, -12,  -3,  -1,  18, -14,  -19,
            -105, -21, -58, -33, -17, -28, -19,  -23,
        ]

        mg_bishop_table = [
            -29,   4, -82, -37, -25, -42,   7,  -8,
            -26,  16, -18, -13,  30,  59,  18, -47,
            -16,  37,  43,  40,  35,  50,  37,  -2,
            -4,   5,  19,  50,  37,  37,   7,  -2,
            -6,  13,  13,  26,  34,  12,  10,   4,
            0,  15,  15,  15,  14,  27,  18,  10,
            4,  15,  16,   0,   7,  21,  33,   1,
            -33,  -3, -14, -21, -13, -12, -39, -21,
        ]

        mg_rook_table = [
            32,  42,  32,  51, 63,  9,  31,  43,
            27,  32,  58,  62, 80, 67,  26,  44,
            -5,  19,  26,  36, 17, 45,  61,  16,
            -24, -11,   7,  26, 24, 35,  -8, -20,
            -36, -26, -12,  -1,  9, -7,   6, -23,
            -45, -25, -16, -17,  3,  0,  -5, -33,
            -44, -16, -20,  -9, -1, 11,  -6, -71,
            -19, -13,   1,  17, 16,  7, -37, -26,
        ]

        mg_queen_table = [
            -28,   0,  29,  12,  59,  44,  43,  45,
            -24, -39,  -5,   1, -16,  57,  28,  54,
            -13, -17,   7,   8,  29,  56,  47,  57,
            -27, -27, -16, -16,  -1,  17,  -2,   1,
            -9, -26,  -9, -10,  -2,  -4,   3,  -3,
            -14,   2, -11,  -2,  -5,   2,  14,   5,
            -35,  -8,  11,   2,   8,  15,  -3,   1,
            -1, -18,  -9,  10, -15, -25, -31, -50,
        ]

        mg_king_table = [
            -65,  23,  16, -15, -56, -34,   2,  13,
            29,  -1, -20,  -7,  -8,  -4, -38, -29,
            -9,  24,   2, -16, -20,   6,  22, -22,
            -17, -20, -12, -27, -30, -25, -14, -36,
            -49,  -1, -27, -39, -46, -44, -33, -51,
            -14, -14, -22, -46, -44, -30, -15, -27,
            1,   7,  -8, -64, -43, -16,   9,   8,
            -15,  36,  12, -54,   8, -28,  24,  14,
        ]

        if piece[1] == 'P':  # Pawn
            if piece[0] == 'w':  # White pawn
                table_value = mg_pawn_table[row * 8 + col]
            else:  # Black pawn
                table_value = mg_pawn_table[(7 - row) * 8 + col]  # Flip rows for black

        elif piece[1] == 'B':  # Bishop
            if piece[0] == 'w': # White bishop
                table_value = mg_bishop_table[row * 8 + col]
            else:
                table_value = mg_bishop_table[(7 - row) * 8 + col]  # Flip rows for black
        
        elif piece[1] == 'N':  # Knight
            if piece[0] == 'w': # White knight
                table_value = mg_knight_table[row * 8 + col]
            else:
                table_value = mg_knight_table[(7 - row) * 8 + col]  # Flip rows for black

        elif piece[1] == 'K':  # King
            if piece[0] == 'w': # White king
                table_value = mg_king_table[row * 8 + col]
            else:
                table_value = mg_king_table[(7 - row) * 8 + col]  # Flip rows for black

        elif piece[1] == 'R': # Rook
            if piece[0] == 'w': # White rook
                table_value = mg_rook_table[row * 8 + col]
            else:
                table_value = mg_rook_table[(7 - row) * 8 + col]  # Flip rows for black
        
        elif piece[1] == 'Q': # Queen
            if piece[0] == 'w': # White queen
                table_value = mg_queen_table[row * 8 + col]
            else:
                table_value = mg_queen_table[(7 - row) * 8 + col]  # Flip rows for black

        return table_value

    def _evaluate_board(self, board):
        """
        Evaluates the board position for the AI.
        Args:
            board (ChessBoard): The current game board.
        Returns:
            int: The evaluation score (positive for advantage, negative for disadvantage).
        """
        pieces_count = sum(len(positions) for positions in board.piece_positions.values())
        is_endgame = board.is_endgame

        mg_piece_values = {'P': 95, 'N': 337, 'B': 365, 'R': 477, 'Q': 1025, 'K': 0}
        eg_piece_values = {'P': 155, 'N': 281, 'B': 297, 'R': 512, 'Q': 936, 'K': 0}
        
        score = 0
        
        for piece, positions in board.piece_positions.items():
            if is_endgame:
                piece_value = eg_piece_values.get(piece[1], 0)
            else:
                piece_value = mg_piece_values.get(piece[1], 0)
            for pos in positions:
                row, col = pos
                table_value = 0

                if is_endgame: # Endgame
                    table_value = self._endgame_positional_value(piece, row, col)
                else:
                    table_value = self._middlegame_positional_value(piece, row, col)

                # Add table value and piece value to score
                if piece[0] == 'w':  # White pieces
                    score -= piece_value + table_value
                else:  # Black pieces
                    score += piece_value + table_value

        return score