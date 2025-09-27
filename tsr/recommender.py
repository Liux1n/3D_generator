import os
from PIL import Image
import imagehash
import clip
import torch

class Recommender:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.image_hashes = {}
        self.image_clip_features = {}
        # self._load_hashes()
        self._load_clip_features()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.recomment_threshold = 0.8

    # def _compute_phash(self, image_path):
    #     """
    #     Calculates the perceptual hash (pHash) of an image and returns an ImageHash object
    #     """
    #     try:
    #         img = Image.open(image_path).convert("L")  
    #         return imagehash.phash(img)
    #     except Exception as e:
    #         return None
    
    def _compute_clip_feature(self, image):
        """
        Calculates the CLIP feature of an image and returns a tensor
        """
        preprocessed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(preprocessed_image)
            
        return image_features
    
    def _compute_phash(self, image):
        """
        Calculates the perceptual hash (pHash) of an image and returns an ImageHash object
        """
        try:
            # return imagehash.phash(image)
            return imagehash.phash(image, hash_size=16)
        except Exception as e:
            return None

    def _load_hashes(self):
        """
        Preload pHash with all preprocessed images
        """
        print("Loading dataset pHash values...")
        
        # print('self.dataset_path', self.dataset_path)
        for item_id in os.listdir(self.dataset_path):
            item_dir = os.path.join(self.dataset_path, item_id)

            preprocess_path = os.path.join(item_dir, "preprocessed.jpg")

            # load image from preprocess_path
            clip_feature_path = os.path.join(item_dir, "clip_feature.pt")

            if os.path.exists(preprocess_path):
                preprocessed_img = Image.open(preprocess_path)

                h = self._compute_phash(preprocessed_img)

                if h is not None:
                    self.image_hashes[item_id] = h
                    
    def _load_clip_features(self):
        """
        Preload CLIP features with all preprocessed images
        """
        print("Loading dataset CLIP features...")
        
        for item_id in os.listdir(self.dataset_path):
            item_dir = os.path.join(self.dataset_path, item_id)
            clip_feature_path = os.path.join(item_dir, "clip_feature.pt")

            if os.path.exists(clip_feature_path):
                clip_feature = torch.load(clip_feature_path)
                self.image_clip_features[item_id] = clip_feature
                
    def recommend_models_clip(self, query_image, top_k=10):
        """
        return the top_k most similar 3D model paths based on CLIP features
        """
        query_clip_feature = self._compute_clip_feature(query_image)
        if query_clip_feature is None:
            raise ValueError("Query image could not be processed!")
        # print('query_clip_feature', query_clip_feature.shape) # query_clip_feature torch.Size([1, 512])
        similarities = []
        candidate_paths = []
        for item_id, feature in self.image_clip_features.items():
            # print('item_id', item_id, 'feature', feature.shape) # feature torch.Size([1, 512])
            similarity = torch.cosine_similarity(query_clip_feature, feature)
            model_path = os.path.join(self.dataset_path, item_id, "3d_model.obj")
            if os.path.exists(model_path):
                similarities.append((item_id, similarity.item(), model_path))


        # sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        print('similarities', similarities)
        
        similarity_scores = [s[1] for s in similarities]
        model_paths = [s[2] for s in similarities]   


        # return [model_path for _,_, model_path in similarities[:top_k]]
        return similarity_scores[:top_k], model_paths[:top_k]
    

    def recommend_models(self, query_image, top_k=10):
        """
        return the top_k most similar 3D model paths
        """
        query_hash = self._compute_phash(query_image)
        image_features = self._compute_clip_feature(query_image)
        print('image_features', image_features)
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