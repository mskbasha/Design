import os
import logging
import cv2
from moviepy.editor import VideoFileClip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/reduce resolution logs.log",
    filemode="a",
)


def resize_video(input_video, output_video, width=224, height=224):
    clip = VideoFileClip(input_video)
    resized_clip = clip.resize((width, height))
    resized_clip.write_videofile(output_video)


videos = [
    x
    for x in os.listdir("/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset")
    if x.endswith(".mp4")
]
videos_downloaded = [
    x
    for x in os.listdir("/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset_224x224")
    if x.endswith(".mp4")
]
from tqdm.notebook import tqdm

for video in tqdm(videos):
    if video in videos_downloaded:
        continue
    logging.info(f"Started processing {video}\n")
    try:
        resize_video(
            f"/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset/{video}",
            f"/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset_224x224/{video}",
        )
        logging.info("Processing done")
    except Exception as e:
        logging.info(f"failed processing -- {e}")
