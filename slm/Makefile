.PHONY: download-games

download-games:
	curl -L -o data/raw/Nakamura.zip https://www.pgnmentor.com/players/Nakamura.zip
	cd data/raw && unzip -o Nakamura.zip
	rm data/raw/Nakamura.zip

instruction-dataset:
	uv run python scripts/generate_instructions.py \
		--hugging-face-dataset-name ridoo14/NakamuraInstruct \

#############
BASE_MODEL = unsloth/LFM2-350M
#############

fine-tuning:
	uv run modal run -m src.instruct.main --model-name $(BASE_MODEL) --invalidate-dataset-cache

#############
CHECKPOINT := LFM2-350M-r16-20250909-141911/checkpoint-10000

evaluate:
	uv run modal run -m src.instruct.evaluate --model-checkpoint $(CHECKPOINT)

merge-model:
	uv run python scripts/generate_merged_model.py \
		--modal-volume-name model_checkpoints \
		--remote-checkpoint-dir $(CHECKPOINT) \
		--local-merged-models-dir ./merged-models


lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .
