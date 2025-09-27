import os
import uuid
import shutil

def save_to_dataset(input_image, edition_text, preprocessed_image, model_paths, root_dir="dataset"):
    """
    Saves the input, description, preprocessed image, and generated model to the dataset database.
    Parameters:
    input_image: PIL.Image (original input image)
    edition_text: str (user input description)
    preprocessed_image: PIL.Image (preprocessed image)
    model_paths: list[str] (model paths (obj, glb))
    root_dir: Database root directory
    Returns:
    save_dir: Path to the saved folder
    """
    # 1. generate unique ID for inout image
    unique_id = str(uuid.uuid4())[:8]
    save_dir = os.path.join(root_dir, unique_id)
    os.makedirs(save_dir, exist_ok=True)

    # 2. save description.txt (including ID and Edition)
    with open(os.path.join(save_dir, "description.txt"), "w", encoding="utf-8") as f:
        f.write(f"ID: {unique_id}\n")
        f.write(f"Edition: {edition_text if edition_text else 'N/A'}\n")

    # 3. save input image
    if input_image is not None:
        input_image.save(os.path.join(save_dir, "input.jpg"))

    # 4. save preprocess image
    if preprocessed_image is not None:
        preprocessed_image.save(os.path.join(save_dir, "preprocessed.jpg"))

    # 5. save 3D model (obj / glb)
    for path in model_paths:
        if path.endswith(".obj"):
            shutil.copy(path, os.path.join(save_dir, "model.obj"))
        elif path.endswith(".glb"):
            shutil.copy(path, os.path.join(save_dir, "model.glb"))

    return save_dir
