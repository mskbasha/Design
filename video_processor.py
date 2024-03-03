import random
from typing import List, Tuple, Union

from tqdm import tqdm
import cv2
import librosa
import numpy as np
import torch
import whisper

random.seed(42)


class VideoProcessor:
    def __init__(
        self,
        text_model,
        image_model,
        text_tokenizer,
        image_processor,
        audio_model="base",
        text_batch_size=32,
        device=torch.device("cuda:0"),
        chunk_size=2,
    ) -> None:
        """Module to process videos.

        Args:
            text_model (torch.nn.Module): model to process text
            image_model (torch.nn.Module): model to process image
            text_tokenizer (_type_): text tokenizer
            image_processor (_type_): image tokenizer
            audio_model (str, optional): whisper model for audio to text. Defaults to 'base'.
            training (bool): parameter to set if text and image models are to be trained
        """
        self.text_model = text_model
        self.image_model = image_model
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.audio_model = whisper.load_model("base")
        self.training = image_model.training or text_model.training
        self.text_batch_size = text_batch_size
        self.device = device
        self.chunk_size = chunk_size

    def __call__(self, video_loc: str) -> torch.tensor:
        frames, audios = self.extract_frames_and_audio(video_loc)
        self.check_models()
        encoded_frames = self.vision_encoder(frames)
        encoded_text = self.audio_encoder(audios)
        return encoded_frames, encoded_text

    def audio_encoder(self, audios):
        text = self.extract_text(audios)
        encoded_text = []
        print("Encoding Text")
        for text_sample in tqdm(text):
            tokenized_text = self.text_tokenizer(text_sample, return_tensors="pt").to(
                self.device
            )
            encoded_text.append(self.text_model(**tokenized_text).pooler_output)
        print("Encoding text completed")
        return torch.stack(encoded_text)

    def vision_encoder(self, frames):
        print("Encoding frames")
        encoded_frames = []
        with torch.inference_mode():
            for frame_batch in tqdm(frames):
                processed_frames = self.image_processor(
                    frame_batch, return_tensors="pt"
                ).to(self.device)
                encoded_frame = self.image_model(**processed_frames).pooler_output.mean(
                    axis=0
                )
                encoded_frames.append(encoded_frame)

        print("Encoding frames done")
        return torch.stack(encoded_frames)

    def check_models(self):
        if self.training:
            self.image_model = self.image_model.train()
            self.text_model = self.text_model.train()
        else:
            self.image_model = self.image_model.eval()
            self.text_model = self.text_model.eval()

    def extract_frames_and_audio(
        self,
        video_path: str,
    ) -> Tuple[Union[torch.tensor, List[np.array]]]:
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

        frames = []
        audio_clips = []
        print(f"Extracting {self.chunk_size} sec batches")
        for i in tqdm(
            range(
                0,
                frame_count - frame_rate * self.chunk_size,
                frame_rate * self.chunk_size,
            )
        ):
            clip_frames = []
            frame_indices = random.sample(
                range(i, i + frame_rate * self.chunk_size), frames_per_clip
            )
            for index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = cap.read()
                if not success:
                    break
                clip_frames.append(cv2.resize(frame, (225, 225)))
            if len(clip_frames) == frames_per_clip:
                frames.append(clip_frames)
                start_frame = i
                end_frame = i + frame_rate * 2
                start_audio = int(start_frame * len(audio_data) / frame_count)
                end_audio = int(end_frame * len(audio_data) / frame_count)
                audio_clip = audio_data[start_audio:end_audio]
                audio_clips.append(audio_clip)

        print(f"Extracting {self.chunk_size} sec batches complete")
        cap.release()
        # Convert frames array to tensor
        converted_clips = torch.tensor(frames)

        return converted_clips, audio_clips

    def extract_text(self, audios: List[np.array]):
        """Method to process audios and covnevrt to text.

        Args:
            audios (List[np.array]): list of audios in numpy array

        Returns:
            List[str]: list of text converted from audio
        """
        text = []
        for audio in audios:
            text.append(self.audio_model.transcribe(audio)["text"])
        return text
