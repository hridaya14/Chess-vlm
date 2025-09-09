from transformers import pipeline

pipe = pipeline("text-generation", model="ridoo14/ChessInstruct-Nakamura")

CHESS_PROMPT_TEMPLATE = """
You are the great Hikaru Nakamura.
Your task is to make the best move in the given game state.

Game state: rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1
Last 5 moves: ['d2d4']
Valid moves: ['g8h6', 'g8f6', 'b8c6', 'b8a6', 'h7h6', 'g7g6', 'f7f6', 'e7e6', 'd7d6', 'c7c6', 'b7b6', 'a7a6', 'h7h5', 'g7g5', 'f7f5', 'e7e5', 'd7d5', 'c7c5', 'b7b5', 'a7a5']

Your next move should be in UCI format (e.g., 'e2e4', 'f8c8').
Make sure your next move is one of the valid moves.
You should only respond with the move in UCI format, 1 word that is the move
"""

generator = pipeline(
    "text-generation", model="ridoo14/ChessInstruct-Nakamura", device="cuda")
output = generator([{"role": "user", "content": CHESS_PROMPT_TEMPLATE}],
                   max_new_tokens=32, return_full_text=False)[0]
print(output["generated_text"])
