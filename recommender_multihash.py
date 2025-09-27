import os
from PIL import Image
import imagehash

class MultiHashRecommender:
    def __init__(self, dataset_path="dataset", phash_size=16, use_dhash=True):
        """
        Initialize the recommender with dataset path and hashing options.

        """
        self.dataset_path = dataset_path
        self.phash_size = phash_size
        self.use_dhash = use_dhash
        self.image_hashes = {}
        self._load_hashes()

    def _compute_multihash(self, image_path):
        """
        Compute a combined hash of pHash + (optional) dHash.

        """
        try:
            img = Image.open(image_path).convert("L")

            # pHash part (length = phash_size^2 bits)
            ph = imagehash.phash(img, hash_size=self.phash_size)
            phash_bits = bin(int(str(ph), 16))[2:].zfill(self.phash_size ** 2)

            # dHash part (length = 64 bits)
            if self.use_dhash:
                dh = imagehash.dhash(img)
                dhash_bits = bin(int(str(dh), 16))[2:].zfill(64)
                return phash_bits + dhash_bits
            else:
                return phash_bits

        except Exception:
            return None

    def _hamming_distance(self, h1, h2):
        """
        Compute the Hamming distance between two binary strings.

        """
        return sum(ch1 != ch2 for ch1, ch2 in zip(h1, h2))

    def _load_hashes(self):
        """
        Preload all preprocessed images' combined hashes from the dataset.
        """
        for item_id in os.listdir(self.dataset_path):
            item_dir = os.path.join(self.dataset_path, item_id)
            preprocess_path = os.path.join(item_dir, "preprocess.jpg")
            if os.path.exists(preprocess_path):
                h = self._compute_multihash(preprocess_path)
                if h is not None:
                    self.image_hashes[item_id] = h

    def recommend_models(self, query_image_path, top_k=10):
        """
        Recommend the most similar 3D models for a given query image.
        """
        query_hash = self._compute_multihash(query_image_path)
        if query_hash is None:
            return []

        distances = []
        for item_id, h in self.image_hashes.items():
            dist = self._hamming_distance(query_hash, h)
            model_path = os.path.join(self.dataset_path, item_id, "3d_model.obj")
            if os.path.exists(model_path):
                distances.append((dist, model_path))

        # Sort by distance ascending (smaller distance = more similar)
        distances.sort(key=lambda x: x[0])

        # Return only model paths
        return [model_path for _, model_path in distances[:top_k]]
