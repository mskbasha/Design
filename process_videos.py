import torch
import pickle
from tqdm import tqdm
import time
import pickle
import torch
import os
from video_processor import VideoProcessor
from transformers import CLIPProcessor, CLIPModel


print("loading model")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("loaded Model")
vp = VideoProcessor(
    clip_model.text_model,
    clip_model.vision_model,
    processor.tokenizer,
    processor.image_processor,
    clip_model.text_projection,
    clip_model.visual_projection,
    device="cpu",
)
vp = vp.to("cuda:0")

vp.device = torch.device("cuda:0")

os.getcwd()
video_dir = "/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset_224x224"
videos = [x for x in os.listdir(video_dir) if x.endswith(".mp4")]

try:
    with open("video_mae_data.pkl", "rb") as f:
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
                with open("video_mae_data.pkl", "wb") as f:
                    pickle.dump(encoded_data, f)
                f1.write("processing Done\n")
            except Exception as e:
                f1.write("Processing Failed {e}\n")
