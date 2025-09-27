import os
from PIL import Image
import imagehash
import clip
import torch


dataset_path = "./dataset"

new_dataset_path = "./new_dataset"
# create new_dataset_path if not exists
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
def compute_clip_feature(image):
    """
    Calculates the CLIP feature of an image and returns a tensor
    """
    preprocessed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_image)
        
    return image_features
# get all folders in dataset_path
all_folders = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print('all_folders', all_folders)
for folder in all_folders:
    image_path = os.path.join(folder, "preprocessed.jpg")
    # replace \ with /
    image_path = image_path.replace("\\", "/")
    image = Image.open(image_path).convert("L")
    # replace \ with /
    
    # print('image', image)
    clip_feature = compute_clip_feature(image)
    print('clip_feature', clip_feature.shape)
    # save clip_feature to folder as clip_feature.pt

    torch.save(clip_feature, os.path.join(folder, "clip_feature.pt"))