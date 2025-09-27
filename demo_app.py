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
from tsr.recommender import Recommender
# import lpips
from functools import partial



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
        
        self.top_k = 5
        self.similar_image_paths = []
        self.instruction = 'N/A'
        self.candidate_models = []
        self.model_buttons = []
        self.curr_obj = None
        self.num_candidates = 0
        self.root_dir="./dataset"
        self.top_match_threshold = 0.95
        self.do_remove_background = True
        self.foreground_ratio = 0.85
        self.mc_resolution = 256
        self.save_dir = None
        
        self.recommender = Recommender(dataset_path=self.root_dir)
    
    
    def save_to_dataset(self, input_image, edition_text, preprocessed_image, unique_id, save_dir):
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


        return self.save_dir
    
    def get_score(self, label):
        mapping = {
            "很差": 1,
            "不太好": 2,
            "一般": 3,
            "好": 4,
            "非常好": 5
        }
        score = mapping.get(label, 0)
        return score
            

    def preprocess(self, input_image, instruction):
        print('instruction:', instruction)
        demo_path = './examples/demo.jpeg'
        # load demo image
        demo_image = Image.open(demo_path).convert("RGBA")
        input_image = demo_image
        
        def fill_background(image):
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            return Image.fromarray((image * 255.0).astype(np.uint8))
        do_remove_background = True
        foreground_ratio = 0.85
        if do_remove_background:
            image = input_image.convert("RGB")
            image = remove_background(image, self.rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = fill_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = fill_background(image)
        
        # demo_pic = ....
        # self.similar_image_paths = self.recommender.recommend_models(image, top_k=self.top_k)
        
        # self.similar_image_paths = self.recommender.recommend_models_clip(image, top_k=self.top_k)
        
        # similarities.append((item_id, similarity.item(), model_path))
        self.similarities, self.similar_image_paths = self.recommender.recommend_models_clip(image, top_k=self.top_k)
        # self.candidates = self.recommender.recommend_models_clip(image, top_k=self.top_k)
        
        self.num_candidates = len(self.similar_image_paths)
        
        
        
        # distances [('821c88c2', 24, './dataset\\821c88c2\\3d_model.obj'), ('bbf40d91', 28, './dataset\\bbf40d91\\3d_model.obj'), ('21a0c4f8', 32, 
        # './dataset\\21a0c4f8\\3d_model.obj'), ('ffe420bc', 36, './dataset\\ffe420bc\\3d_model.obj')]
        # print('similar_image_paths:', self.similar_image_paths)
        # self.most_relative_models = self.find_most_relative_models()
        return image


    def generate_mesh(self, image, mc_resolution):
        
        paths = []
        
        root_dir = self.root_dir
        # 1. generate unique ID for inout image
        self.unique_id = str(uuid.uuid4())[:8]
        self.save_dir = os.path.join(root_dir, self.unique_id)
        self.save_dir = self.save_dir.replace("\\", "/")
  
        if self.similarities[0] >= self.top_match_threshold:
            paths.append(self.similar_image_paths[0])
            self.curr_model_path = self.similar_image_paths[0]
            # paths.append(self.curr_model_path)
            # return paths
            return self.similar_image_paths[0]
        else:
            scene_codes = self.model(image, device=self.device)
            mesh = self.model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
            mesh = to_gradio_3d_orientation(mesh)
            mesh_path = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False)
            mesh.export(mesh_path.name)
            paths.append(mesh_path.name)
            save_dir = "./saved_models/"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename('temp_model.obj'))
                    
            # shutil.copy(mesh_path.name, save_path)  
            
            self.curr_model_path = save_path
            # paths.append(save_path)
            return mesh_path.name
            
            
            # for fmt in formats:
            #     mesh_path = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
            #     mesh.export(mesh_path.name)
            #     paths.append(mesh_path.name)
            #     if fmt == "obj":
            #         save_dir = "./saved_models/"
            #         os.makedirs(save_dir, exist_ok=True)
            #         save_path = os.path.join(save_dir, os.path.basename('temp_model.obj'))
                    
            #         shutil.copy(mesh_path.name, save_path)  
            #         self.curr_model_path = save_path
                    
            #         # self.curr_model_path = 
            
            # paths.append(save_path)
        # for path in self.similar_image_paths:
        #     paths.append(path)
        
        # return paths
    

  
    # def update_candidates_button(self):
    #     return [gr.update(
            
    
    def update_candidates(self):

        # new_candidates = [gr.update(value=path) for path in self.similar_image_paths]
        # new_candidates = [gr.update(value=path, label=f"相似度： {self.similarities[i]:.4f}") for i, path in enumerate(self.similar_image_paths)]
        new_candidates = [gr.update(value=path, label = f"相似度： {self.similarities[i] * 100:.2f}%") for i, path in enumerate(self.similar_image_paths)]
        return new_candidates
            
    def set_main_model(path):
        return path

    def get_curr_model_path(self):
        # print('get_curr_model_path:', self.curr_model_path)
        return self.curr_model_path
    
    def process_review(self, choice):
        return f"你选择的评分是：{choice}"
    
    
    def save_to_db(self, choice):

        score = self.get_score(choice)
        
        if self.save_dir is not None:
            # save_dir = self._check_save_dir(self.save_dir)


            score_file = os.path.join(self.save_dir, "score.txt")
            score_file = score_file.replace("\\", "/")
            os.makedirs(os.path.dirname(score_file), exist_ok=True)

            with open(score_file, "w", encoding="utf-8") as f:
                f.write(f"{score}\n")
            print(f"Saved score to {score_file}")
            
            gr.Info("✅ 评分提交成功！谢谢你的反馈！")
        else:
            gr.Info("❌ 请生成3D模型后再提交评分！")
        # return "✅ 评分已提交，谢谢！"
    
    def build_interface(self):
        with gr.Blocks(title="3D模型生成") as interface:
            gr.Markdown(
                """
                # 3D Generator Demo
                Single image → 3D mesh reconstruction.
                """
            )

            with gr.Row(variant="panel"):
                # left column: input image and parameters
                with gr.Column(scale=0.45):
                    with gr.Row():
                        with gr.Column():
                            self.instruction = gr.Textbox(lines=1, label="请输入提示词", interactive=True)
                        # with gr.Column(scale=0.15):
                        #     self.example_btn = gr.Button("确认", variant="secondary")
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
                            # do_remove_background = gr.Checkbox(label="Remove Background", value=True)
     
                            # foreground_ratio = gr.Slider(0.5, 1.0, value=0.85, step=0.05, label="Foreground Ratio")
                            mc_resolution = gr.Slider(32, 320, value=256, step=32, label="3D模型分辨率")
      
                    with gr.Row():
                        submit = gr.Button("开始生成", variant="primary")

                # right column: output 3D model and candidates
                with gr.Column(scale=0.55):
                    with gr.Row():
                        with gr.Column(scale=0.5):
                            with gr.Row():
                                with gr.Tab("生成结果"):
                                    self.output_model_obj = gr.Model3D(label="OBJ 格式模型预览", interactive=False, height=400)
                            with gr.Column():
                                review = gr.Radio(
                                    label="你觉得模型生成质量如何？",
                                    choices=['差', '一般', '好', '非常好'],
                                    interactive=True
                                )
                                # result = gr.Textbox(label="结果", interactive=False)
                                
                                submit_btn = gr.Button("提交评分")
                                # result = gr.Textbox(label="", interactive=False)  # 可选：显示“提交成功”提示
                                # 当用户选择后调用 process_review
                                submit_btn.click(self.save_to_db, inputs=review, outputs=None)
                                                                    

                        with gr.Column(scale=0.5):
                            with gr.Tab("你可能感兴趣的模型"):
                                with gr.Column(elem_id="scrollable-col"):

                                        
                                    for i in range(self.top_k):
                                        m = gr.Model3D(
                                            # label=f"相似度： {self.similarities[i]:.4f}",
                                            label=None,
                                            # value=None,
                                            height=300,
                                            interactive=False
                                        )
                                        self.candidate_models.append(m)
                    
            
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


            submit.click(
                fn=lambda img, txt: self._check_input(img, txt),
                inputs=[input_image, self.instruction],
            ).success(
                fn=self.preprocess,
                # inputs=[input_image, do_remove_background, foreground_ratio],
                inputs=[input_image, self.instruction],
                outputs=processed_image,
            ).success(
                fn=self.update_candidates,
                inputs=None,
                outputs=self.candidate_models
            ).success(
                fn=self.generate_mesh,
                inputs=[processed_image, mc_resolution],
                outputs=self.output_model_obj,
            )
            # .success(
            #     fn=self.save_to_dataset,
            #     # save_to_dataset(self, input_image, edition_text, preprocessed_image, model_paths, self.unique_id, self.save_dir):
            #     inputs=[input_image, self.instruction, processed_image],
            #     outputs=[],
            # )

        return interface

    # --------- 工具函数 ---------
    @staticmethod
    def _check_input(img, text):
        if img is None:
            raise gr.Error("❌ 请先上传图片!")
        if text == 'N/A' or text.strip() == "":
            gr.Info("⚠️ 未输入提示词，将使用原始图片进行生成！")
            return img, None
        else:
            return img, text.strip()
    
    @staticmethod
    def _check_save_dir(save_dir):
        if save_dir is None:
            raise gr.Error("❌ 请先生成3D模型！")
        return save_dir

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