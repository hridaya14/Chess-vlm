from transformers import pipeline
from jinja2 import Template

CHESS_PROMPT_TEMPLATE = """
You are the great Hikaru Nakamura.
Your task is to make the best move in the given game state.

Game state: {{ game_state }}
Last 5 moves: {{ last_5_moves_uci }}
Valid moves: {{ valid_moves }}

Your next move should be in UCI format (e.g., 'e2e4', 'f8c8').
Make sure your next move is one of the valid moves.
"""


def get_prompt(
    # player_to_move: str,
    game_state: str,
    last_5_moves_uci: list[str],
    valid_moves: list[str],
) -> str:
    template = Template(CHESS_PROMPT_TEMPLATE)
    prompt = template.render(
        # player_to_move=player_to_move,
        game_state=game_state,
        last_5_moves_uci=last_5_moves_uci,
        valid_moves=valid_moves,
    )
    return prompt


game_state = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
last_5_moves_uci = ["d2d4"]
valid_moves = [
    "g8h6",
    "g8f6",
    "b8c6",
    "b8a6",
    "h7h6",
    "g7g6",
    "f7f6",
    "e7e6",
    "d7d6",
    "c7c6",
    "b7b6",
    "a7a6",
    "h7h5",
    "g7g5",
    "f7f5",
    "e7e5",
    "d7d5",
    "c7c5",
    "b7b5",
    "a7a5"
]

prompt = get_prompt(game_state, last_5_moves_uci, valid_moves)

generator = pipeline(
    "text-generation", model="ridoo14/ChessInstruct-Nakamura", device="cuda")
output = generator([{"role": "user", "content": prompt}],
                   max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
