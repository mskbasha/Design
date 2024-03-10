import os

import cv2
from moviepy.editor import VideoFileClip


def resize_video(input_video, output_video, width=224, height=224):
    clip = VideoFileClip(input_video)
    resized_clip = clip.resize((width, height))
    resized_clip.write_videofile(output_video)


videos = [
    x
    for x in os.listdir("/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset")
    if x.endswith(".mp4")
]
from tqdm.notebook import tqdm

with open("reduce_resolution.txt", "a") as f:
    for video in tqdm(videos):
        f.write(f"Started processing {video}\n")
        try:
            resize_video(
                f"/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset/{video}",
                f"/DATA/sarmistha_2221cs21/basha/VideoMAE/dataset_224x224/{video}",
            )
            f.write("Processing done")
        except:
            f.write("failed processing")
