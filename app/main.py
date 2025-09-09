import sys
import chess
import chess.svg
import cairosvg
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import pipeline


# === Load Model ===
MODEL_NAME = "ridoo14/ChessInstruct-Nakamura"
pipe = pipeline("text-generation", model=MODEL_NAME)
generator = pipeline(
    # Change to cpu incase of no gpu
    "text-generation", model=MODEL_NAME, device="cuda")
print(f"Loading model: {MODEL_NAME}")
# === Game State ===
board = chess.Board()


def board_to_qpixmap(board: chess.Board, size: int) -> QPixmap:
    """Render chess board into a QPixmap for display in PyQt."""
    svg_data = chess.svg.board(board=board, size=size)
    png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
    qimg = QImage.fromData(png_data)
    return QPixmap.fromImage(qimg)


def get_llm_move() -> str | None:
    """Ask the LLM for the next move in UCI format."""
    last_moves = [m.uci() for m in list(board.move_stack)[-5:]]
    valid_moves = [m.uci() for m in board.legal_moves]

    prompt = f"""
    You are the great Hikaru Nakamura
    Your task is to make the best move in the given game state.
    Game state: {board.fen()}
    Last 5 moves: {last_moves}
    Valid moves: {valid_moves}

    Your next move should be in UCI format (e.g., 'e2e4', 'f8c8').
    Make sure your next move is one of the valid moves.
    You should only respond with the move in UCI format, 1 word that is the move
    """
    response = generator([{"role": "user", "content": prompt}],
                         max_new_tokens=32, return_full_text=False)[0]
    print("#########################")
    print(f"{response}")
    print("#########################")

    for move in valid_moves:
        if move in response:
            return move

    import random
    if valid_moves:
        return random.choice(valid_moves)
    return None


class ChessApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess vs LLM")
        self.resize(900, 950)

        self.selected_square = None

        # === Layout ===
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Player indicators
        self.turn_layout = QHBoxLayout()
        self.you_label = QLabel("You (White)")
        self.llm_label = QLabel("LLM (Black)")
        self.you_label.setAlignment(Qt.AlignCenter)
        self.llm_label.setAlignment(Qt.AlignCenter)
        self.turn_layout.addWidget(self.you_label)
        self.turn_layout.addWidget(self.llm_label)
        self.layout.addLayout(self.turn_layout)

        # Chessboard Display
        self.board_label = QLabel()
        self.board_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.board_label)

        # Moves display
        self.moves_label = QLabel("Moves:\n")
        self.moves_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.moves_label)

        # Enable clicks
        self.board_label.mousePressEvent = self.on_board_click

        self.render_board()
        self.update_turn_indicator()

    def render_board(self):
        size = min(self.width(), self.height()) - 150
        pixmap = board_to_qpixmap(board, size)
        self.board_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        self.render_board()

    def update_turn_indicator(self):
        if board.turn == chess.WHITE:
            self.you_label.setStyleSheet("color: green; font-weight: bold;")
            self.llm_label.setStyleSheet("color: black;")
        else:
            self.you_label.setStyleSheet("color: black;")
            self.llm_label.setStyleSheet("color: red; font-weight: bold;")

    def on_board_click(self, event):
        size = min(self.width(), self.height()) - 150
        square_size = size / 8
        x = event.pos().x() - (self.board_label.width() - size) / 2
        y = event.pos().y() - (self.board_label.height() - size) / 2

        if 0 <= x < size and 0 <= y < size:
            file = int(x // square_size)
            rank = 7 - int(y // square_size)
            square = chess.square(file, rank)

            if self.selected_square is None:
                self.selected_square = square
            else:
                move = chess.Move(self.selected_square, square)
                self.selected_square = None
                self.make_player_move(move)

    def make_player_move(self, move: chess.Move):
        if move in board.legal_moves:
            board.push(move)
            self.render_board()
            self.append_move_text("You", move.uci())
            self.update_turn_indicator()
            QApplication.processEvents()
            self.llm_turn()

    def llm_turn(self):
        move = get_llm_move()
        if move:
            board.push_uci(move)
            self.render_board()
            self.append_move_text("LLM", move)
            self.update_turn_indicator()

    def append_move_text(self, player: str, move: str):
        current_text = self.moves_label.text()
        new_text = current_text + f"{player}: {move}\n"
        self.moves_label.setText(new_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChessApp()
    window.show()
    sys.exit(app.exec_())
