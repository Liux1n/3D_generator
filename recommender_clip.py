import os
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

class CLIPRecommender:
    def __init__(self, dataset_path="dataset", device=None):

        self.dataset_path = dataset_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Storage for features {item_id: embedding vector}
        self.image_features = {}
        self._load_features()

    def _extract_feature(self, image_path):
        """
        Extract CLIP embedding for a single image.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            img_input = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model.encode_image(img_input)

            # Normalize the feature vector
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu()
        except Exception:
            return None

    def _load_features(self):
        """
        Preload CLIP embeddings for all preprocessed images in dataset.
        """
        for item_id in os.listdir(self.dataset_path):
            item_dir = os.path.join(self.dataset_path, item_id)
            preprocess_path = os.path.join(item_dir, "preprocess.jpg")
            if os.path.exists(preprocess_path):
                feat = self._extract_feature(preprocess_path)
                if feat is not None:
                    self.image_features[item_id] = feat

        # Stack into a feature matrix for faster retrieval
        self.ids = list(self.image_features.keys())
        self.feats = torch.cat([self.image_features[i] for i in self.ids], dim=0)

    def recommend_models(self, query_image_path, top_k=10):
        """
        Recommend the most similar 3D models for a given query image using CLIP.
        
        """
        query_feat = self._extract_feature(query_image_path)
        if query_feat is None:
            return []

        # Compute cosine similarity
        sims = cosine_similarity(query_feat.numpy(), self.feats.numpy())[0]

        # Get top_k indices
        top_indices = sims.argsort()[::-1][:top_k]

        # Return corresponding model paths
        results = []
        for idx in top_indices:
            item_id = self.ids[idx]
            model_path = os.path.join(self.dataset_path, item_id, "3d_model.obj")
            if os.path.exists(model_path):
                results.append(model_path)

        return results
