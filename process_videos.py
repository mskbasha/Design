from PIL import Image
import torch
import pickle
from tqdm import tqdm
import requests
import pickle
import torch
import os
from video_processor import VideoProcessor
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
vp = VideoProcessor(
    clip_model.text_model,
    clip_model.vision_model,
    processor.tokenizer,
    processor.image_processor,
    clip_model.text_projection,
    clip_model.visual_projection,
    device="cpu",
)
vp = vp.to("cuda:2")

vp.device = torch.device("cuda:2")

os.getcwd()
video_dir = "/home/basha_2211ai03/complaint_detection/videos/dataset"
videos = os.listdir(video_dir)

try:
    with open("data.pkl", "rb") as f:
        encoded_data = pickle.load(f)
except:
    encoded_data = {}

with torch.inference_mode():
    for video in tqdm(videos):
        with open("output.txt", "a") as f1:
            f1.write(f"Processing {video} started\t\t")
            if video in encoded_data:
                continue
            try:
                out = vp(os.path.join(video_dir, video))
                encoded_data[video] = out
                with open("data.pkl", "wb") as f:
                    pickle.dump(encoded_data, f)
                f1.write("processing Done\n")
            except:
                f1.write("Processing Failed\n")
