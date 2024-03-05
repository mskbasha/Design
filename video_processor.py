import random
from typing import List, Tuple, Union

from tqdm import tqdm
import cv2
import librosa
import numpy as np
import torch
import whisper

random.seed(42)


class VideoProcessor(torch.nn.Module):
    def __init__(
        self,
        text_model,
        image_model,
        text_tokenizer,
        image_processor,
        text_projection=None,
        image_projection=None,
        audio_model="base",
        text_batch_size=32,
        device=torch.device("cuda:0"),
        chunk_size=2,
    ) -> None:
        super(VideoProcessor, self).__init__()
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
        self.audio_model = whisper.load_model(
            "base",
            device=device,
        )
        self.training = image_model.training or text_model.training
        self.text_batch_size = text_batch_size
        self.device = device
        self.chunk_size = chunk_size
        self.text_projection = text_projection
        self.image_projection = image_projection

    def __call__(self, video_loc: str) -> torch.tensor:
        frames, audios = self.extract_frames_and_audio(video_loc)
        encoded_frames = self.vision_encoder(frames)
        encoded_text = self.audio_encoder(audios)
        return encoded_frames, encoded_text

    def audio_encoder(self, audios: List[np.array]) -> torch.tensor:
        """Method to encode audio given audio in numpy array format.

        Args:
            audios (List[np.array]): list of numpy arrays of sr 16000

        Returns:
            torch.tensor: encoded audio
        """
        text = self.extract_text(audios)
        encoded_text = []
        print("Encoding Text")
        with torch.inference_mode(not self.training):
            for text_sample in tqdm(text):
                tokenized_text = self.text_tokenizer(
                    text_sample, return_tensors="pt"
                ).to(self.device)
                pooler_text = self.text_model(**tokenized_text).pooler_output
                if self.text_projection:
                    pooler_text = self.text_projection(pooler_text)
                encoded_text.append(pooler_text)
        print("Encoding text completed")
        return torch.stack(encoded_text)

    @torch.inference_mode
    def vision_encoder(self, frames: List[np.array]) -> torch.tensor:
        """Method to encode frames

        Args:
            frames (list[np.array]): list of numpy arrays of size 224 x 224

        Returns:
            torch.tensor: encoded frames in
        """
        print("Encoding frames")
        with torch.inference_mode(not self.training):
            encoded_frames = []
            for frame_batch in tqdm(frames):
                processed_frames = self.image_processor(
                    frame_batch, return_tensors="pt"
                ).to(self.device)
                encoded_frames.append(
                    self.image_model(**processed_frames)
                    .last_hidden_state.mean(1)
                    .squeeze()
                )
        print("Encoding frames done")
        return torch.stack(encoded_frames)

    def extract_frames_and_audio(
        self, video_path: str, number_of_frames_to_extract=16
    ) -> Tuple[Union[torch.tensor, List[np.array]]]:
        """Given a video path extracts frames and audio for every 2sec.

        Args:
            video_path (str): location of the video

        Returns:
            list: list of frames and audio data
        """
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Extracting frame chunks")

        def extract_frame(indices: List[int]) -> np.array:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for ind in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.resize(frame, (224, 224)))
            return frames

        frame_indices = []
        frames = []
        for index in tqdm(range(frame_count)):
            frame_indices.append(index)
            if (index + 1) % (self.chunk_size * frame_rate) == 0:
                indices = sorted(
                    random.sample(frame_indices, number_of_frames_to_extract)
                )
                frames.append(extract_frame(indices))
        audio_data, _ = librosa.load(video_path, sr=16000)
        last_ind = 0
        audio_chunk_size = len(audio_data) // len(frames)
        audios = []
        print("Extracting audio chunks")
        for i in tqdm(range(len(frames))):
            audios.append(audio_data[i * audio_chunk_size : (i + 1) * audio_chunk_size])
        return frames, audios

    def extract_text(self, audios: List[np.array]):
        """Method to process audios and convert to text.

        Args:
            audios (List[np.array]): list of audios in numpy array

        Returns:
            List[str]: list of text converted from audio
        """
        print("Extracting text from audio")
        text = []
        for audio in tqdm(audios):
            text.append(self.audio_model.transcribe(audio)["text"])
        print("Extracting text from audio completed")
        return text
