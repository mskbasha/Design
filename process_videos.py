import logging
import os
import pickle
import time

import torch
from tqdm import tqdm
from transformers import (AutoImageProcessor, CLIPModel, CLIPProcessor,
                          VideoMAEModel)

from video_processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/clip.log",
    filemode="a",
)

print("loading model")
logging.info("Loading Model")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
video_mae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
logging.info("Loaded Model")
print("loaded Model")
vp = VideoProcessor(
    clip_model.text_model,
    video_mae,
    processor.tokenizer,
    video_processor,
    text_projection=clip_model.text_projection,
)
vp = vp.to("cuda:0")

vp.device = torch.device("cuda:0")

os.getcwd()
video_dir = "/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset_224x224"
videos = [x for x in os.listdir(video_dir) if x.endswith(".mp4")]

try:
    with open("data/video_mae_data.pkl", "rb") as f:
        encoded_data = pickle.load(f)
except:
    encoded_data = {}
with torch.inference_mode():
    vp = vp.eval()
    for video in tqdm(videos):
        logging.info(("-" * 10) + f"Processing {video} started\t\t" + ("-" * 10))
        if video in encoded_data:
            continue
        try:
            out = vp(os.path.join(video_dir, video))
            encoded_data[video] = out
            with open("data/video_mae_data.pkl", "wb") as f:
                pickle.dump(encoded_data, f)
            logging.info(("-" * 10) + "processing Done\n" + ("-" * 10))
        except Exception as e:
            logging.warning(("X" * 10) + f"Processing Failed {e}\n" + ("X" * 10))
logging.info("#" * 100)
logging.info("Completed processing Videos")
