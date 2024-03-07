import torch
from tqdm import tqdm
import psutil
import pickle
import os
from video_processor import VideoProcessor
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, VideoMAEModel

print("Loading model")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
video_mae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
print("Loaded Model")
vp = VideoProcessor(
    clip_model.text_model,
    video_mae,
    processor.tokenizer,
    video_processor,
    text_projection=clip_model.text_projection,
)
process = psutil.Process()
vp = vp.to("cuda:0")

vp.device = torch.device("cuda:0")

os.getcwd()
video_dir = "/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset_224x224"
videos = [x for x in os.listdir(video_dir) if x.endswith(".mp4")]
try:
    with open("data/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
except:
    dataset = {}
for video in tqdm(videos):
    print("-" * 90)
    print(f"Processing started {video}")
    if video in dataset:
        continue
    try:
        dataset[video] = vp.extract_frames_and_audio(os.path.join(video_dir, video))
    except Exception as e:
        print(f"Processing failed {e}")
    print("Memory usage:", process.memory_info().rss / (1024 * 1024), "MB")
    with open("data/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print(f"Processing Done")
    print("-" * 90)
