import random
from typing import List, Tuple

import cv2
import librosa
import numpy as np

random.seed(42)


def extract_frames_and_audio(video_path: str) -> Tuple[List[np.array]]:
    """Given a video path extracts frames and audio for every 2sec.

    Args:
        video_path (str): location of the video

    Returns:
        list: list of frames and audio data
    """
    cap = cv2.VideoCapture(video_path)
    frames_per_clip = 6
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    audio_data, _ = librosa.load(video_path, sr=16000)

    clips = []
    audio_clips = []

    for i in range(0, frame_count - frame_rate * 2, frame_rate * 2):
        clip_frames = []
        frame_indices = random.sample(range(i, i + frame_rate * 2), frames_per_clip)
        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = cap.read()
            if not success:
                break
            clip_frames.append(cv2.resize(frame, (225, 225)))
        if len(clip_frames) == frames_per_clip:
            clips.append(clip_frames)
            start_frame = i
            end_frame = i + frame_rate * 2
            start_audio = int(start_frame * len(audio_data) / frame_count)
            end_audio = int(end_frame * len(audio_data) / frame_count)
            audio_clip = audio_data[start_audio:end_audio]
            audio_clips.append(audio_clip)

    cap.release()
    return clips, audio_clips


if __name__ == "__main__":
    # Example usage:
    video_path = "VideoMAE/dataset/-AAVugSu0Fw.mp4"
    clips, audio_clips = extract_frames_and_audio(video_path)

    print(f"Number of 2-second clips extracted: {len(clips)}")
    print(f"Number of audio clips extracted: {len(audio_clips)}")
