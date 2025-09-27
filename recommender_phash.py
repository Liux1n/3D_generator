import os
from PIL import Image
import imagehash

class PhashRecommender:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.image_hashes = {}
        self._load_hashes()

    def _compute_phash(self, image_path):
        """
        Calculates the perceptual hash (pHash) of an image and returns an ImageHash object
        """
        try:
            img = Image.open(image_path).convert("L")  
            return imagehash.phash(img)
        except Exception as e:
            return None

    def _load_hashes(self):
        """
        Preload pHash with all preprocessed images
        """
        print("Loading dataset pHash values...")
        for item_id in os.listdir(self.dataset_path):
            item_dir = os.path.join(self.dataset_path, item_id)
            preprocess_path = os.path.join(item_dir, "preprocess.jpg")
            if os.path.exists(preprocess_path):
                h = self._compute_phash(preprocess_path)
                if h is not None:
                    self.image_hashes[item_id] = h

    def recommend_models(self, query_image_path, top_k=10):
        """
        return the top_k most similar 3D model paths
        """
        query_hash = self._compute_phash(query_image_path)
        if query_hash is None:
            raise ValueError("Query image could not be processed!")

        distances = []
        for item_id, h in self.image_hashes.items():
            dist = query_hash - h  
            model_path = os.path.join(self.dataset_path, item_id, "3d_model.obj")
            if os.path.exists(model_path):
                distances.append((item_id, dist, model_path))

        # sort
        distances.sort(key=lambda x: x[1])

        return [model_path for _, model_path in distances[:top_k]]
