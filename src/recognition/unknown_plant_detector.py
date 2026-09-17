import numpy as np
import os
from pathlib import Path
from .clip_embedder import CLIPEmbedder

class UnknownPlantDetector:
    def __init__(self, embeddings_dir="data/embeddings", similarity_threshold=0.75):
        self.clip = CLIPEmbedder()
        self.embeddings_dir = embeddings_dir
        self.similarity_threshold = similarity_threshold
        self.class_embeddings = {}
        self.class_names = []
        
        # Load pre-computed embeddings
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load embeddings from disk"""
        embeddings_path = Path(self.embeddings_dir)
        if embeddings_path.exists():
            for embedding_file in embeddings_path.glob("*.npy"):
                class_name = embedding_file.stem
                self.class_embeddings[class_name] = np.load(embedding_file)
                self.class_names.append(class_name)
            print(f"Loaded {len(self.class_embeddings)} class embeddings")
        else:
            print(f"Warning: embeddings directory {self.embeddings_dir} not found")
    
    def detect(self, image_path):
        """
        Detect if plant is unknown or known.
        Returns: {
            'status': 'known' or 'unknown',
            'class': class_name (if known),
            'confidence': similarity score,
            'all_scores': dict of all class similarities
        }
        """
        image_embedding = self.clip.get_image_embedding(image_path)
        if image_embedding is None:
            return {'status': 'error', 'message': 'Failed to process image'}
        
        # Calculate similarities with all known classes
        similarities = {}
        for class_name, class_embedding in self.class_embeddings.items():
            sim = self.clip.cosine_similarity(image_embedding, class_embedding)
            similarities[class_name] = float(sim)
        
        # Find the best match
        if not similarities:
            return {'status': 'error', 'message': 'No embeddings loaded'}
        
        best_class = max(similarities, key=similarities.get)
        best_score = similarities[best_class]
        
        result = {
            'status': 'known' if best_score >= self.similarity_threshold else 'unknown',
            'class': best_class if best_score >= self.similarity_threshold else None,
            'confidence': best_score,
            'all_scores': similarities,
            'threshold': self.similarity_threshold
        }
        
        return result
    
    def save_class_embedding(self, class_name, image_path):
        """Save embedding for a class (for initialization)"""
        embedding = self.clip.get_image_embedding(image_path)
        if embedding is not None:
            os.makedirs(self.embeddings_dir, exist_ok=True)
            np.save(os.path.join(self.embeddings_dir, f"{class_name}.npy"), embedding)
            print(f"Saved embedding for {class_name}")