import os
import uuid
import shutil

def save_to_dataset(input_image, edition_text, preprocessed_image, model_paths, root_dir="dataset"):
    """
    保存输入、描述、预处理图像、生成模型到 dataset 数据库中
    参数:
        input_image: PIL.Image 输入原图
        edition_text: str 用户输入描述
        preprocessed_image: PIL.Image 预处理图
        model_paths: list[str] 模型路径 (obj, glb)
        root_dir: 数据库根目录
    返回:
        save_dir: 存储的文件夹路径
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

    print(f"[INFO] All data saved in {save_dir}")
    return save_dir
