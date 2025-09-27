import os
from PIL import Image
import imagehash

class PhashRecommender:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.image_hashes = {}
        self._load_hashes()

    # def _compute_phash(self, image_path):
    #     """
    #     Calculates the perceptual hash (pHash) of an image and returns an ImageHash object
    #     """
    #     try:
    #         img = Image.open(image_path).convert("L")  
    #         return imagehash.phash(img)
    #     except Exception as e:
    #         return None
    
    def _compute_phash(self, image):
        """
        Calculates the perceptual hash (pHash) of an image and returns an ImageHash object
        """
        try:
            return imagehash.phash(image)
        except Exception as e:
            return None

    def _load_hashes(self):
        """
        Preload pHash with all preprocessed images
        """
        print("Loading dataset pHash values...")
        
        print('self.dataset_path', self.dataset_path)
        for item_id in os.listdir(self.dataset_path):
            item_dir = os.path.join(self.dataset_path, item_id)
            print('item_dir', item_dir)
            preprocess_path = os.path.join(item_dir, "preprocessed.jpg")
            print('preprocess_path', preprocess_path)
            # load image from preprocess_path
            

            if os.path.exists(preprocess_path):
                preprocessed_img = Image.open(preprocess_path)
                print('preprocessed_img', preprocessed_img)
                h = self._compute_phash(preprocessed_img)
                print('h', h)
                if h is not None:
                    self.image_hashes[item_id] = h

    def recommend_models(self, query_image, top_k=10):
        """
        return the top_k most similar 3D model paths
        """
        query_hash = self._compute_phash(query_image)
        if query_hash is None:
            raise ValueError("Query image could not be processed!")

        distances = []
        for item_id, h in self.image_hashes.items():
            # print('item_id', item_id, 'h', h)
            dist = query_hash - h  
            model_path = os.path.join(self.dataset_path, item_id, "3d_model.obj")
            if os.path.exists(model_path):
                distances.append((item_id, dist, model_path))

        # sort by distance
        distances.sort(key=lambda x: x[1])
        print('distances', distances)
        # distances [('bbf40d91', 34, './dataset\\bbf40d91\\3d_model.obj'), ('ffe420bc', 34, './dataset\\ffe420bc\\3d_model.obj'), ('821c88c2', 40, './dataset\\821c88c2\\3d_model.obj'), ('21a0c4f8', 42, './dataset\\21a0c4f8\\3d_model.obj')]  

        return [model_path for _,_, model_path in distances[:top_k]]