KNIGHT_MOVES = [
    0x0000000000020400, 0x0000000000050800, 0x00000000000A1100, 0x0000000000142200,
    0x0000000000284400, 0x0000000000508800, 0x0000000000A01000, 0x0000000000402000,
    0x0000000002040004, 0x0000000005080008, 0x000000000A110011, 0x0000000014220022,
    0x0000000028440044, 0x0000000050880088, 0x00000000A0100010, 0x0000000040200020,
    0x0000000204000402, 0x0000000508000805, 0x0000000A1100110A, 0x0000001422002214,
    0x0000002844004428, 0x0000005088008850, 0x000000A0100010A0, 0x0000004020002040,
    0x0000020400040200, 0x0000050800080500, 0x00000A1100110A00, 0x0000142200221400,
    0x0000284400442800, 0x0000508800885000, 0x0000A0100010A000, 0x0000402000204000,
    0x0002040004020000, 0x0005080008050000, 0x000A1100110A0000, 0x0014220022140000,
    0x0028440044280000, 0x0050880088500000, 0x00A0100010A00000, 0x0040200020400000,
    0x0204000402000000, 0x0508000805000000, 0x0A1100110A000000, 0x1422002214000000,
    0x2844004428000000, 0x5088008850000000, 0xA0100010A0000000, 0x4020002040000000,
    0x0400040200000000, 0x0800080500000000, 0x1100110A00000000, 0x2200221400000000,
    0x4400442800000000, 0x8800885000000000, 0x100010A000000000, 0x2000204000000000,
    0x0004020000000000, 0x0008050000000000, 0x00110A0000000000, 0x0022140000000000,
    0x0044280000000000, 0x0088500000000000, 0x0010A00000000000, 0x0020400000000000,
]

KING_MOVES = [
    0x0000000000000302, 0x0000000000000705, 0x0000000000000E0A, 0x0000000000001C14,
    0x0000000000003828, 0x0000000000007050, 0x000000000000E0A0, 0x000000000000C040,
    0x0000000000030203, 0x0000000000070507, 0x00000000000E0A0E, 0x00000000001C141C,
    0x0000000000382838, 0x0000000000705070, 0x0000000000E0A0E0, 0x0000000000C040C0,
    0x0000000003020300, 0x0000000007050700, 0x000000000E0A0E00, 0x000000001C141C00,
    0x0000000038283800, 0x0000000070507000, 0x00000000E0A0E000, 0x00000000C040C000,
    0x0000000302030000, 0x0000000705070000, 0x0000000E0A0E0000, 0x0000001C141C0000,
    0x0000003828380000, 0x0000007050700000, 0x000000E0A0E00000, 0x000000C040C00000,
    0x0000030203000000, 0x0000070507000000, 0x00000E0A0E000000, 0x00001C141C000000,
    0x0000382838000000, 0x0000705070000000, 0x0000E0A0E0000000, 0x0000C040C0000000,
    0x0003020300000000, 0x0007050700000000, 0x000E0A0E00000000, 0x001C141C00000000,
    0x0038283800000000, 0x0070507000000000, 0x00E0A0E000000000, 0x00C040C000000000,
    0x0302030000000000, 0x0705070000000000, 0x0E0A0E0000000000, 0x1C141C0000000000,
    0x3828380000000000, 0x7050700000000000, 0xE0A0E00000000000, 0xC040C00000000000,
    0x0203000000000000, 0x0507000000000000, 0x0A0E000000000000, 0x141C000000000000,
    0x2838000000000000, 0x5070000000000000, 0xA0E0000000000000, 0x40C0000000000000,
]

class BitboardChess():
    def __init__(self):
        # Initialize bitboards for each piece type
        self.white_pawns = 0x000000000000FF00
        self.black_pawns = 0x00FF000000000000
        self.white_rooks = 0x0000000000000081
        self.black_rooks = 0x8100000000000000
        self.white_knights = 0x0000000000000042
        self.black_knights = 0x4200000000000000
        self.white_bishops = 0x0000000000000024
        self.black_bishops = 0x2400000000000000
        self.white_queen = 0x0000000000000008
        self.black_queen = 0x0800000000000000
        self.white_king = 0x0000000000000010
        self.black_king = 0x1000000000000000

        # Additional tracking
        self.all_white_pieces = (self.white_pawns | self.white_rooks | self.white_knights |
                                self.white_bishops | self.white_queen | self.white_king)
        self.all_black_pieces = (self.black_pawns | self.black_rooks | self.black_knights |
                                self.black_bishops | self.black_queen | self.black_king)
        self.all_pieces = self.all_white_pieces | self.all_black_pieces
        self.current_turn = 'white'
        self.move_history = []

    
    def display_bitboard(self, bitboard):
        """
        Visualizes a bitboard in an 8x8 chessboard format.
        """
        for rank in range(7, -1, -1):
            line = ""
            for file in range(8):
                square = rank * 8 + file
                line += "1 " if (bitboard >> square) & 1 else ". "
            print(line)
        print()

    def execute_move(self, start, end, promotion=None):
        """
        Executes a move on the bitboard.
        """
        self.move_piece(start, end, self.turn, promotion)
        self.move_history.append((start, end))
        self.turn = 'black' if self.turn == 'white' else 'white'

    def move_piece(self, start, end, color, promotion=None):
        """
        Executes a move on the board and handles special cases (castling, promotion, en passant).
        Args:
            start (int): Starting square (0-63).
            end (int): Ending square (0-63).
            color (str): 'white' or 'black'.
            promotion (str, optional): Piece to promote to ('queen', 'rook', 'bishop', 'knight').
        Returns:
            None
        """
        moving_piece = None
        opponent_pieces = self.all_black_pieces if color == 'white' else self.all_white_pieces

        # Determine which piece is moving
        for piece_type in ['pawns', 'knights', 'bishops', 'rooks', 'queen', 'king']:
            bitboard = getattr(self, f"{color}_{piece_type}")
            if bitboard & (1 << start):
                moving_piece = piece_type
                setattr(self, f"{color}_{piece_type}", bitboard & ~(1 << start) | (1 << end))
                break

        # Handle captures
        for piece_type in ['pawns', 'knights', 'bishops', 'rooks', 'queen', 'king']:
            opponent_bitboard = getattr(self, f"{'black' if color == 'white' else 'white'}_{piece_type}")
            if opponent_bitboard & (1 << end):
                setattr(self, f"{'black' if color == 'white' else 'white'}_{piece_type}", opponent_bitboard & ~(1 << end))
                break

        # Handle special cases
        if moving_piece == 'king' and abs(start - end) == 2:  # Castling
            if end > start:  # Kingside
                rook_start, rook_end = (7, 5) if color == 'white' else (63, 61)
            else:  # Queenside
                rook_start, rook_end = (0, 3) if color == 'white' else (56, 59)
            rook_bitboard = getattr(self, f"{color}_rooks")
            setattr(self, f"{color}_rooks", rook_bitboard & ~(1 << rook_start) | (1 << rook_end))

        elif moving_piece == 'pawns' and (end < 8 or end >= 56):  # Promotion
            promoted_piece = getattr(self, f"{color}_{promotion or 'queen'}")
            setattr(self, f"{color}_{promotion or 'queen'}", promoted_piece | (1 << end))
            pawns_bitboard = getattr(self, f"{color}_pawns")
            setattr(self, f"{color}_pawns", pawns_bitboard & ~(1 << end))

        elif moving_piece == 'pawns':  # En passant
            if abs(start - end) in [7, 9] and not (opponent_pieces & (1 << end)):
                en_passant_square = end + (8 if color == 'white' else -8)
                opponent_pawns = getattr(self, f"{'black' if color == 'white' else 'white'}_pawns")
                setattr(self, f"{'black' if color == 'white' else 'white'}_pawns", opponent_pawns & ~(1 << en_passant_square))

        # Update castling rights
        if moving_piece == 'king':
            setattr(self, f"{color}_king", 0)  # King can no longer castle
        if moving_piece == 'rooks':
            if start in [0, 7]:  # Update rook rights for white
                self.white_rooks &= ~(1 << start)
            elif start in [56, 63]:  # Update rook rights for black
                self.black_rooks &= ~(1 << start)

        # Recalculate piece sets
        self.all_white_pieces = self.white_pawns | self.white_rooks | self.white_knights | self.white_bishops | self.white_queen | self.white_king
        self.all_black_pieces = self.black_pawns | self.black_rooks | self.black_knights | self.black_bishops | self.black_queen | self.black_king
        self.all_pieces = self.all_white_pieces | self.all_black_pieces

        # Switch turn
        self.turn = 'white' if self.turn == 'black' else 'black'

    def capture_piece(self, start, end, moving_bitboard, opponent_bitboard):
        moving_bitboard = self.move_piece(start, end, moving_bitboard)
        opponent_bitboard &= ~(1 << end)  # Remove captured piece
        return moving_bitboard, opponent_bitboard
    
    def generate_moves_for_piece(self, piece_bitboard, piece_type, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn):
        """
        Generates moves for a specific piece type.
        """
        if piece_type == 'pawn':
            return self.generate_pawn_moves(piece_bitboard, own_pieces, opponent_pieces, all_pieces, color == 'white', king_bitboard, is_check_fn)
        elif piece_type == 'knight':
            return self.generate_knight_moves(piece_bitboard, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn)
        elif piece_type == 'bishop':
            return self.generate_bishop_moves(piece_bitboard, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn)
        elif piece_type == 'rook':
            return self.generate_rook_moves(piece_bitboard, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn)
        elif piece_type == 'queen':
            return self.generate_queen_moves(piece_bitboard, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn)
        elif piece_type == 'king':
            return self.generate_king_moves(piece_bitboard, own_pieces, opponent_pieces, all_pieces, color, is_check_fn)
        return []
    
    def generate_legal_moves(self, color, own_pieces, opponent_pieces, king_bitboard, all_pieces, is_check_fn):
        """
        Generates all legal moves for the current player.
        """
        piece_types = {
            'pawn': self.white_pawns if color == 'white' else self.black_pawns,
            'knight': self.white_knights if color == 'white' else self.black_knights,
            'bishop': self.white_bishops if color == 'white' else self.black_bishops,
            'rook': self.white_rooks if color == 'white' else self.black_rooks,
            'queen': self.white_queen if color == 'white' else self.black_queen,
            'king': self.white_king if color == 'white' else self.black_king,
        }

        all_moves = []
        for piece_type, piece_bitboard in piece_types.items():
            all_moves.extend(self.generate_moves_for_piece(
                piece_bitboard, piece_type, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn
            ))

        return all_moves

    def generate_pawn_moves(self, pawns, own_pieces, opponent_pieces, all_pieces, is_white, king_bitboard, is_check_fn):
        """
        Generates legal pawn moves for a given side.
        """
        moves = []
        if is_white:
            single_push = (pawns << 8) & ~all_pieces
            double_push = ((pawns & 0x000000000000FF00) << 16) & ~all_pieces & ~(all_pieces << 8)
            captures_left = (pawns << 7) & opponent_pieces & ~0x8080808080808080
            captures_right = (pawns << 9) & opponent_pieces & ~0x0101010101010101
        else:
            single_push = (pawns >> 8) & ~all_pieces
            double_push = ((pawns & 0x00FF000000000000) >> 16) & ~all_pieces & ~(all_pieces >> 8)
            captures_left = (pawns >> 9) & opponent_pieces & ~0x0101010101010101
            captures_right = (pawns >> 7) & opponent_pieces & ~0x8080808080808080

        possible_moves = single_push | double_push | captures_left | captures_right
        while possible_moves:
            target = possible_moves & -possible_moves
            end = target.bit_length() - 1
            start = pawns & (1 << (end - 8 if is_white else end + 8))
            if start:
                start_square = start.bit_length() - 1
                if self.is_move_legal(start_square, end, "white" if is_white else "black", all_pieces, own_pieces, opponent_pieces, king_bitboard, is_check_fn):
                    moves.append((start_square, end))
            possible_moves &= possible_moves - 1
        return moves
    
    def generate_knight_moves(self, knights, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn):
        """
        Generates legal knight moves.
        """
        moves = []
        while knights:
            knight_pos = knights & -knights  # Isolate LSB
            square = knight_pos.bit_length() - 1
            targets = KNIGHT_MOVES[square] & ~own_pieces  # Filter self-captures
            for end in range(64):
                if targets & (1 << end):
                    if self.is_move_legal(square, end, color, all_pieces, own_pieces, opponent_pieces, king_bitboard, is_check_fn):
                        moves.append((square, end))
            knights &= knights - 1  # Remove processed knight
        return moves

    def generate_king_moves(self, king, own_pieces, opponent_pieces, all_pieces, color, is_check_fn):
        """
        Generates legal king moves, including castling.
        """
        moves = []
        square = king.bit_length() - 1
        targets = KING_MOVES[square] & ~own_pieces  # Exclude self-captures

        # Standard King Moves
        for end in range(64):
            if targets & (1 << end):
                if self.is_move_legal(square, end, color, all_pieces, own_pieces, opponent_pieces, king, is_check_fn):
                    moves.append((square, end))

        # Add Castling Moves
        king_moved = self.white_king != 0x0000000000000010 if color == 'white' else self.black_king != 0x1000000000000000
        rook_moved = {
            0: self.white_rooks & 0x0000000000000001 == 0,  # a1
            7: self.white_rooks & 0x0000000000000080 == 0,  # h1
            56: self.black_rooks & 0x0100000000000000 == 0,  # a8
            63: self.black_rooks & 0x8000000000000000 == 0,  # h8
        }
        castling_moves = self.can_castle(color, king_moved, rook_moved, all_pieces, is_check_fn)
        moves.extend(castling_moves)

        return moves

    
    def generate_bishop_moves(self, bishops, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn):
        """
        Generates legal bishop moves using sliding bitboard logic.
        """
        moves = []
        while bishops:
            bishop_pos = bishops & -bishops
            square = bishop_pos.bit_length() - 1
            directions = [-9, -7, 7, 9]  # Diagonal directions
            for direction in directions:
                target = square + direction
                while 0 <= target < 64:
                    if (all_pieces & (1 << target)):
                        if (opponent_pieces & (1 << target)):  # Capture
                            if self.is_move_legal(square, target, color, all_pieces, own_pieces, opponent_pieces, king_bitboard, is_check_fn):
                                moves.append((square, target))
                        break
                    if self.is_move_legal(square, target, color, all_pieces, own_pieces, opponent_pieces, king_bitboard, is_check_fn):
                        moves.append((square, target))
                    target += direction
            bishops &= bishops - 1
        return moves
    
    def generate_rook_moves(self, rooks, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn):
        """
        Generates legal rook moves using sliding bitboard logic.
        """
        moves = []
        while rooks:
            rook_pos = rooks & -rooks
            square = rook_pos.bit_length() - 1
            directions = [-8, 8, -1, 1]  # Horizontal and vertical directions
            for direction in directions:
                target = square + direction
                while 0 <= target < 64:
                    if (all_pieces & (1 << target)):
                        if (opponent_pieces & (1 << target)):  # Capture
                            if self.is_move_legal(square, target, color, all_pieces, own_pieces, opponent_pieces, king_bitboard, is_check_fn):
                                moves.append((square, target))
                        break
                    if self.is_move_legal(square, target, color, all_pieces, own_pieces, opponent_pieces, king_bitboard, is_check_fn):
                        moves.append((square, target))
                    target += direction
            rooks &= rooks - 1
        return moves

    def generate_queen_moves(self, queens, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn):
        """
        Generates legal queen moves.
        """
        bishop_moves = self.generate_bishop_moves(queens, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn)
        rook_moves = self.generate_rook_moves(queens, own_pieces, opponent_pieces, all_pieces, color, king_bitboard, is_check_fn)
        return bishop_moves + rook_moves


    def is_move_legal(self, start, end, color, all_pieces, own_pieces, opponent_pieces, king_bitboard, is_check_fn):
        """
        Checks if a move is legal.
        Args:
            start (int): Starting square index (0-63).
            end (int): Ending square index (0-63).
            color (str): 'white' or 'black'.
            all_pieces (int): Bitboard of all pieces.
            own_pieces (int): Bitboard of the player's pieces.
            opponent_pieces (int): Bitboard of the opponent's pieces.
            king_bitboard (int): Bitboard of the player's king.
            is_check_fn (callable): Function to check if the king is in check.
        Returns:
            bool: True if the move is legal, False otherwise.
        """
        # 1. Ensure move is within bounds
        if not (0 <= end < 64):
            return False

        # 2. Ensure no self-capture
        if own_pieces & (1 << end):
            return False

        # 3. Ensure the move does not leave the king in check
        new_own_pieces = (own_pieces & ~(1 << start)) | (1 << end)
        new_all_pieces = (all_pieces & ~(1 << start)) | (1 << end)
        if is_check_fn(king_bitboard, new_own_pieces, opponent_pieces, new_all_pieces):
            return False

        return True
    
    def is_check(self, king_bitboard, own_pieces, opponent_pieces, all_pieces, move_start=None, move_end=None):
        """
        Checks if the king is in check by any of the opponent's pieces.
        Args:
            king_bitboard (int): Bitboard representing the king's position.
            own_pieces (int): Bitboard of all the player's pieces.
            opponent_pieces (dict): Dictionary of bitboards for opponent's pieces (e.g., 'knights', 'rooks').
            all_pieces (int): Bitboard of all pieces on the board.
            move_start (int, optional): The starting square of the temporary move.
            move_end (int, optional): The ending square of the temporary move.
        Returns:
            bool: True if the king is in check, False otherwise.
        """
        opponent_pieces_copy = {piece: bitboard for piece, bitboard in opponent_pieces.items()}
        # Temporarily update the bitboards if a move is specified
        if move_start is not None and move_end is not None:
            moving_piece = (own_pieces & (1 << move_start))

            # Update own_pieces
            own_pieces = (own_pieces & ~(1 << move_start)) | (1 << move_end)
            
            # Update all_pieces
            all_pieces = (all_pieces & ~(1 << move_start)) | (1 << move_end)

            # If capturing an opponent's piece, remove it from the opponent_pieces
            for piece_type, bitboard in opponent_pieces_copy.items():
                if bitboard & (1 << move_end):
                    opponent_pieces_copy[piece_type] &= ~(1 << move_end)

        king_position = king_bitboard.bit_length() - 1

        # Check for knight attacks
        if self._knight_attacks(king_position) & opponent_pieces_copy['knights']:
            return True

        # Check for rook/queen attacks (horizontal/vertical)
        if self._rook_attacks(king_position, all_pieces) & (opponent_pieces_copy['rooks'] | opponent_pieces_copy['queens']):
            return True

        # Check for bishop/queen attacks (diagonal)
        if self._bishop_attacks(king_position, all_pieces) & (opponent_pieces_copy['bishops'] | opponent_pieces_copy['queens']):
            return True

        # Check for pawn attacks
        if self._pawn_attacks(king_position, opponent_pieces_copy['pawns'], is_white=(king_bitboard & own_pieces) > 0):
            return True

        # Check for king attacks
        if self._king_attacks(king_position) & opponent_pieces_copy['kings']:
            return True

        return False

    def _knight_attacks(self, position):
        """
        Returns a bitboard of knight attacks from a given position.
        """
        return KNIGHT_MOVES[position]

    def _rook_attacks(self, position, all_pieces):
        """
        Returns a bitboard of rook attacks from a given position, accounting for obstructions.
        """
        return self._sliding_attacks(position, all_pieces, directions=[-8, 8, -1, 1])

    def _bishop_attacks(self, position, all_pieces):
        """
        Returns a bitboard of bishop attacks from a given position, accounting for obstructions.
        """
        return self._sliding_attacks(position, all_pieces, directions=[-9, -7, 7, 9])

    def _sliding_attacks(self, position, all_pieces, directions):
        """
        Returns a bitboard of sliding attacks (rook/bishop/queen) from a given position.
        """
        attacks = 0
        for direction in directions:
            target = position + direction
            while 0 <= target < 64:
                # Break if crossing board boundaries
                if direction in [-1, 1] and (target // 8 != position // 8):
                  break  # Prevent wrapping across rows
                  
                attacks |= (1 << target)
                if all_pieces & (1 << target):  # Stop if obstructed
                    break

                target += direction
        return attacks

    def _pawn_attacks(self, position, pawns, is_white):
        """
        Returns a bitboard of pawn attacks on the king's position.
        """
        if is_white:
            return ((1 << (position - 9)) & ~0x8080808080808080) | ((1 << (position - 7)) & ~0x0101010101010101)
        else:
            return ((1 << (position + 9)) & ~0x0101010101010101) | ((1 << (position + 7)) & ~0x8080808080808080)

    def _king_attacks(self, position):
        """
        Returns a bitboard of king attacks from a given position.
        """
        return KING_MOVES[position]
    
    def can_castle(self, color, king_moved, rook_moved, all_pieces, is_check_fn):
        """
        Determines if castling is legal for the given player.
        """
        if color == 'white':
            king_position = 4  # e1
            kingside_rook = 7  # h1
            queenside_rook = 0  # a1
            empty_kingside = 0x0000000000000060  # f1, g1
            empty_queenside = 0x000000000000000E  # b1, c1, d1
        else:
            king_position = 60  # e8
            kingside_rook = 63  # h8
            queenside_rook = 56  # a8
            empty_kingside = 0x6000000000000000  # f8, g8
            empty_queenside = 0x0E00000000000000  # b8, c8, d8

        castling_moves = []

        # Kingside castling
        if not king_moved and not rook_moved[kingside_rook]:
            if not (all_pieces & empty_kingside):
                if not is_check_fn(1 << king_position, self.all_white_pieces if color == 'white' else self.all_black_pieces, 
                                   self.all_black_pieces if color == 'white' else self.all_white_pieces, all_pieces):
                    castling_moves.append((king_position, king_position + 2))

        # Queenside castling
        if not king_moved and not rook_moved[queenside_rook]:
            if not (all_pieces & empty_queenside):
                if not is_check_fn(1 << king_position, self.all_white_pieces if color == 'white' else self.all_black_pieces, 
                                   self.all_black_pieces if color == 'white' else self.all_white_pieces, all_pieces):
                    castling_moves.append((king_position, king_position - 2))

        return castling_moves