from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os


def _as_feature_tensor(output):
    """
    Newer versions of `transformers` (5.x) can return a BaseModelOutputWithPooling
    object from get_image_features/get_text_features instead of a plain tensor.
    This normalizes either case into a plain tensor.
    """
    if torch.is_tensor(output):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unexpected output type from CLIP feature extraction: {type(output)}")


class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"CLIP loaded on {device}")

    def get_image_embedding(self, image_path):
        """Extract embedding from a single image"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = _as_feature_tensor(image_features)
            # Normalize
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def get_text_embedding(self, text):
        """Extract embedding from text"""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = _as_feature_tensor(text_features)
        embedding = text_features / text_features.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0]

    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
