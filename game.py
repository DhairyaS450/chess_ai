import pygame
from chess_board import ChessBoard

class GameController:
    def __init__(self):
        """
        Initializes the game controller, including the chessboard and pygame.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))  # 480x480 pixel window
        pygame.display.set_caption("Chess Game")
        self.clock = pygame.time.Clock()

        # Chessboard and game state
        self.board = ChessBoard()
        self.selected_piece = None  # Currently selected piece (row, col)
        self.legal_moves = []       # Legal moves for the selected piece
        self.running = True         # Main game loop flag

        # Load piece images
        self.piece_images = self._load_piece_images()

    def _load_piece_images(self):
        """
        Loads images for chess pieces and returns a dictionary mapping pieces to images.
        """
        images = {}
        piece_types = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
        for piece in piece_types:
            images[piece] = pygame.image.load(f'assets/pieces/{piece}.png')
        return images

    def draw_board(self):
        """
        Draws the chessboard and all pieces.
        """
        colors = [pygame.Color(240, 217, 181), pygame.Color(181, 136, 99)]  # Light and dark squares
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * 60, row * 60, 60, 60))

                # Draw piece if present
                piece = self.board.board[row][col]
                if piece != '..':
                    self.screen.blit(self.piece_images[piece], (col * 60, row * 60))

    def highlight_moves(self):
        """
        Highlights the selected piece and its legal moves.
        """
        if self.selected_piece:
            row, col = self.selected_piece
            pygame.draw.rect(self.screen, pygame.Color(0, 255, 0, 100),
                             pygame.Rect(col * 60, row * 60, 60, 60), 5)  # Highlight selected piece

            for move in self.legal_moves:
                _, end_pos = move
                r, c = end_pos
                pygame.draw.circle(self.screen, pygame.Color(0, 255, 0),
                                   (c * 60 + 30, r * 60 + 30), 10)  # Highlight possible moves

    def handle_click(self, pos):
        """
        Handles mouse clicks to select pieces and make moves.
        Args:
            pos (tuple): (x, y) position of the mouse click in pixels.
        """
        col, row = pos[0] // 60, pos[1] // 60  # Convert pixel coordinates to board indices

        if self.selected_piece:  # A piece is already selected
            move = (self.selected_piece, (row, col))
            if move in self.legal_moves:  # Valid move
                self.board.execute_move(move)
                self.selected_piece = None
                self.legal_moves = []
            else:  # Invalid move, deselect
                self.selected_piece = None
                self.legal_moves = []
        else:  # No piece is selected
            piece = self.board.board[row][col]
            if piece.startswith(self.board.turn[0]):  # Select a piece of the current player
                self.selected_piece = (row, col)
                self.legal_moves = self.board._get_piece_moves((row, col), piece)

    def game_loop(self):
        """
        Main game loop for handling events and rendering the game.
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())

            # Draw the board and update the screen
            self.draw_board()
            self.highlight_moves()
            pygame.display.flip()
            self.clock.tick(60)  # Limit to 60 FPS

        pygame.quit()

game = GameController()
game.__init__()
game.game_loop()