import logging
import os
import tempfile
import time
import glob
import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial
import os
import uuid
import shutil
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
from PIL import Image
import imagehash
import argparse
import shutil
from tsr.recommender_phash import PhashRecommender
import lpips

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, mc_resolution, formats=["obj", "glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv

def find_most_relative_models(input_image, top_k=3):
    # This is a placeholder function. Replace with actual implementation.
    # For demonstration, we return a list of example model paths.
    candidates_path = "./example_models/"
    # DEBUG: randomly select top_k models from the directory
    candidates = glob.glob(os.path.join(candidates_path, "*.obj"))
    # print("candidates:", candidates)
    # DEBUG: randomly select top_k models from the directory
    example_models = candidates[:top_k]
    return example_models

def generate_example(image, mc_resolution, formats=["obj", "glb"]):

    rv = []
    example_path_obj = "./examples/burger.obj"
    example_path_glb = "./examples/burger.glb"

    rv.append(example_path_obj)
    rv.append(example_path_glb)
    return rv


def generate_obj(image, mc_resolution, formats=["obj", "glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    # for format in formats:
    #     print("format:", format)
    #     mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
    #     mesh.export(mesh_path.name)
    #     rv.append(mesh_path.name)
    
    for format in formats:
            mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
            mesh.export(mesh_path.name)
            if format == "obj":
                rv.append(mesh_path.name)
    return rv


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj, mesh_name_glb = generate(preprocessed, 256, ["obj", "glb"])
    return preprocessed, mesh_name_obj, mesh_name_glb


def run_example_obj(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj = generate_obj(preprocessed, 256, ["obj"])
    return preprocessed, mesh_name_obj


class TripoSRDemo:
    def __init__(self, device=None):
  
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Using device: {self.device}")

        print("Loading TripoSR model...")
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(8192)
        self.model.to(self.device)
        print("Model loaded.")
        self.curr_model_path = None
        # ---- rembg 初始化 ----
        self.rembg_session = rembg.new_session()
        
        
        self.similar_image_paths = []
        self.instruction = 'N/A'
        
        self.curr_obj = None
        
        self.root_dir="./dataset"
        
        self.recommender = PhashRecommender(dataset_path=self.root_dir)
    
    
    def save_to_dataset(self, input_image, edition_text, preprocessed_image):
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
        root_dir = self.root_dir
        # 1. generate unique ID for inout image
        unique_id = str(uuid.uuid4())[:8]
        save_dir = os.path.join(root_dir, unique_id)
        save_dir = save_dir.replace("\\", "/")
        # save_dir = os.path.abspath(save_dir)
        # save_dir = os.path.join(root_dir, unique_id)
        # replace \ with /
        # save_dir = save_dir.replace("\", "/")
        # print('save_dir:', save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving data to {save_dir}...")

        # 2. save description.txt (including ID and Edition)
        with open(os.path.join(save_dir, "description.txt"), "w", encoding="utf-8") as f:
            f.write(f"ID: {unique_id}\n")
            f.write(f"Edition: {edition_text if edition_text else 'N/A'}\n")

        # 3. save input image
        # print('input_image:', input_image)
        # if isinstance(input_image, np.ndarray):
        #     print("input_image 是 numpy array")
        # elif isinstance(input_image, Image.Image):
        #     print("input_image 是 PIL.Image")
        # else:
        #     print("input_image 是其他类型:", type(input_image))
        
        input_image = Image.fromarray(np.array(input_image))
        
        if input_image is not None:
            if input_image.mode == "RGBA":
                input_image = input_image.convert("RGB")
            input_image.save(os.path.join(save_dir, "input.jpg"))
            print(f"Saved input image to {os.path.join(save_dir, 'input.jpg')}")

        # print('preprocessed_image:', type(preprocessed_image))
        preprocessed_image = Image.fromarray(np.array(preprocessed_image))
        # 4. save preprocess image
        if preprocessed_image is not None:
            if preprocessed_image.mode == "RGBA":
                preprocessed_image = preprocessed_image.convert("RGB")
            preprocessed_image.save(os.path.join(save_dir, "preprocessed.jpg"))
            print(f"Saved preprocessed image to {os.path.join(save_dir, 'preprocessed.jpg')}")

        # 5. save 3D model (obj / glb)

        shutil.copy(self.curr_model_path, os.path.join(save_dir, "3d_model.obj"))
        print(f"Saved 3D model to {os.path.join(save_dir, '3d_model.obj')}")


        return save_dir
        
    # --------- 图像预处理 ---------
    def preprocess(self, input_image, do_remove_background, foreground_ratio):
        def fill_background(image):
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            return Image.fromarray((image * 255.0).astype(np.uint8))

        if do_remove_background:
            image = input_image.convert("RGB")
            image = remove_background(image, self.rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = fill_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = fill_background(image)
                
        self.similar_image_paths = self.recommender.recommend_models(image, top_k=5)
        # distances [('821c88c2', 24, './dataset\\821c88c2\\3d_model.obj'), ('bbf40d91', 28, './dataset\\bbf40d91\\3d_model.obj'), ('21a0c4f8', 32, 
        # './dataset\\21a0c4f8\\3d_model.obj'), ('ffe420bc', 36, './dataset\\ffe420bc\\3d_model.obj')]
        # print('similar_image_paths:', self.similar_image_paths)
        # self.most_relative_models = self.find_most_relative_models()
        return image


    def generate_mesh(self, image, mc_resolution, formats=["obj", "glb"]):
        scene_codes = self.model(image, device=self.device)
        mesh = self.model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
        mesh = to_gradio_3d_orientation(mesh)

        paths = []
        for fmt in formats:
            mesh_path = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
            mesh.export(mesh_path.name)
            paths.append(mesh_path.name)
            if fmt == "obj":
                save_dir = "./saved_models/"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, os.path.basename('temp_model.obj'))
                shutil.copy(mesh_path.name, save_path)  
                self.curr_model_path = save_path
                
                # self.curr_model_path = 
        paths.append(save_path)
        return paths
    

  
    def find_most_relative_models(self):
        # candidates_path = "./example_models/"
        # candidates = glob.glob(os.path.join(candidates_path, "*.obj"))
        
        return self.similar_image_paths
    
    def update_curr_model(self):
        self.curr_model = gr.Model3D(
                    label="生成模型",
                    value=self.curr_model_path,
                    height=150,
                    interactive=False,
                )
        
    def set_main_model(path):
        return path

    def get_curr_model_path(self):
        # print('get_curr_model_path:', self.curr_model_path)
        return self.curr_model_path
    
    # def 
    
    
    def build_interface(self):
        with gr.Blocks(title="TripoSR") as interface:
            gr.Markdown(
                """
                # 3D Generator Demo
                Single image → 3D mesh reconstruction.
                """
            )

            with gr.Row(variant="panel"):
                # left column: input image and parameters
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=0.85):
                            self.instruction = gr.Textbox(lines=1, label="请输入提示词", interactive=True)
                        with gr.Column(scale=0.15):
                            self.example_btn = gr.Button("确认", variant="secondary")
                            # self.example_btn.click(
                            #     fn=lambda: "a 3D model of a burger",
                            #     inputs=None,
                            #     outputs=self.instruction,
                            # )
                    with gr.Row():
                        input_image = gr.Image(label="Input Image", type="pil", image_mode="RGBA")
                        processed_image = gr.Image(label="Processed Image", interactive=False) # numpy array
                        # processed_image = Image.fromarray(processed_image)
                        
                        
                        
                    with gr.Row():
                        with gr.Group():
                            do_remove_background = gr.Checkbox(label="Remove Background", value=True)
                            foreground_ratio = gr.Slider(0.5, 1.0, value=0.85, step=0.05, label="Foreground Ratio")
                            mc_resolution = gr.Slider(32, 320, value=256, step=32, label="Marching Cubes Resolution")
                    with gr.Row():
                        submit = gr.Button("Generate", variant="primary")

                # right column: output 3D model and candidates
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            with gr.Tab("OBJ"):
                                self.output_model_obj = gr.Model3D(label="Output Model (OBJ)", interactive=False, height=300)
                            with gr.Tab("GLB"):
                                self.output_model_glb = gr.Model3D(label="Output Model (GLB)", interactive=False, height=300)

          
                            
                            model_buttons = []

                        with gr.Column():
                            # self.curr_model_path = './saved_models/figure.obj'
                            # print('self.curr_model_path:', self.curr_model_path)
                            with gr.Column(elem_id="scrollable-col"):
                                self.curr_obj = gr.Model3D(
                                    label="生成模型",
                                    # value=self.curr_model_path,
                                    height=150,
                                    interactive=False,
                                )
            
                                btn = gr.Button(f"展示")
                                # model_buttons.append((btn, self.curr_model_path))
                                btn.click(
                                    fn=lambda: self.curr_model_path,
                                    inputs=None,
                                    outputs=self.output_model_obj
                                )
                                if self.similar_image_paths:
                                    for i, path in enumerate(self.similar_image_paths):
                                        gr.Model3D(label=f"候选模型 {i+1}", value=path, height=150, interactive=False)
                                        btn = gr.Button(f"设为主模型 {i+1}")
                                        model_buttons.append((btn, path))
                                        
                                    gr.HTML("""
                                                <style>
                                                #scrollable-col {
                                                    max-height: 600px;
                                                    overflow-y: auto;
                                                    border: 1px solid #eee;
                                                    padding: 8px;
                                                }
                                                </style>
                                            """)

                    
                                    for btn, path in model_buttons:
                                        btn.click(
                                            fn=lambda p=path: p,
                                            inputs=None,
                                            outputs=self.output_model_obj
                                        )
                                
                                
                                
        #     def save_to_dataset(self, input_image, edition_text, preprocessed_image, model_paths, root_dir="dataset"):
        # """
        # Saves the input, description, preprocessed image, and generated model to the dataset database.
        # Parameters:
        # input_image: PIL.Image (original input image)
        # edition_text: str (user input description)
        # preprocessed_image: PIL.Image (preprocessed image)
        # model_paths: list[str] (model paths (obj, glb))
        # root_dir: Database root directory
        # Returns:
        # save_dir: Path to the saved folder
        # """

            submit.click(
                fn=lambda img: self._check_input(img),
                inputs=[input_image]
            ).success(
                fn=self.preprocess,
                inputs=[input_image, do_remove_background, foreground_ratio],
                outputs=processed_image,
            ).success(
                fn=self.generate_mesh,
                inputs=[processed_image, mc_resolution],
                outputs=[self.output_model_obj, self.output_model_glb, self.curr_obj],
            ).success(
                # [MODIFIED] save all data to dataset
                # fn=lambda processed, obj, glb, inp, edition: self.save_to_dataset(
                #     inp, edition, processed, [obj, glb]
                # ),
                fn=self.save_to_dataset,
                # save_to_dataset(self, input_image, edition_text, preprocessed_image, model_paths, root_dir="./dataset"):
                inputs=[input_image, self.instruction, processed_image],
                outputs=[],
            )

        return interface

    # --------- 工具函数 ---------
    @staticmethod
    def _check_input(img):
        if img is None:
            raise gr.Error("❌ No image uploaded!")
        return img

    # --------- 启动入口 ---------
    def launch(self, **kwargs):
        interface = self.build_interface()
        interface.queue(max_size=kwargs.get("queuesize", 1))
        interface.launch(
            auth=(kwargs.get("username"), kwargs.get("password")) if kwargs.get("username") else None,
            share=kwargs.get("share", False),
            server_name="0.0.0.0" if kwargs.get("listen") else None,
            server_port=kwargs.get("port", 7860)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--queuesize", type=int, default=1)
    args = parser.parse_args()

    demo = TripoSRDemo()
    demo.launch(
        username=args.username,
        password=args.password,
        port=args.port,
        listen=args.listen,
        share=args.share,
        queuesize=args.queuesize,
    )