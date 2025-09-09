#!/usr/bin/env python3
"""
Script to merge LoRA adapter checkpoints with base model and push to Hugging Face Hub
"""

import os

import torch
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder, login


def push_model_to_hf(
    hub_model_name: str = None,
    hub_token: str = None,
    model_path: str = None
):
    """
    Push merged model to hugging face

    Args:
        hub_model_name: Name for the model on HF Hub (e.g., "username/model-name")
        hub_token: Hugging Face token for authentication
        model_path: Path to the merged model
    """

    print("🔄 Starting Upload Process...")

    if hub_model_name:
        print("🚀 Pushing to Hugging Face Hub...")

        # Login to Hugging Face
        if hub_token:
            login(token=hub_token)
        else:
            print("🔑 Please login to Hugging Face (you'll be prompted)...")
            login()

        create_repo(hub_model_name, private=False)

        upload_folder(
            repo_id=hub_model_name,
            folder_path=model_path,   # path to your model folder
            commit_message="Upload trained model",
            path_in_repo=""
        )
        print(
            f"🎉 Model successfully pushed to https://huggingface.co/{
                hub_model_name}"
        )


# Example usage
if __name__ == "__main__":

    HUB_MODEL_NAME = "ridoo14/Fischer-ChessInstruct"  # HF Hub model name

    MERGED_MODEL_PATH = "/workspace/Projects/vlm-chess-engine/vlm/merged-models/LFM2-700M-r16-20250907-132155"

    HF_TOKEN = os.getenv("HF_TOKEN")  # Set this or use interactive login

    try:
        # Merge the adapter
        push_model_to_hf(
            hub_model_name=HUB_MODEL_NAME,
            hub_token=HF_TOKEN,
            model_path=MERGED_MODEL_PATH
        )
        print("🎊 All done! Your merged model is ready to use.")

    except Exception as e:
        print(f"❌ Error during merge process: {str(e)}")
        raise
