"""Script to run preprocessing pipeline."""

from video_processors import HandTrackingProcessor
from mediapipe import MediapipeTarget
from dataset_mapper import DatasetMapper
from pathlib import Path
from typing import List
import os
import cv2


if __name__ == "__main__":
    mediapipe_path = Path('/home/rlashof/repos/mediapipe')
    processor = HandTrackingProcessor(
        MediapipeTarget.MULTI_HAND_TRACKING_CPU,
        False,
        mediapipe_path)

    repo_path = '/home/rlashof/repos/speach-to-sign-language'
    raw_videos_path = f'{repo_path}/raw_videos_mp4'
    out_videos_path = f'{repo_path}/processed_videos'
    if (not os.path.exists(out_videos_path)):
        os.mkdir(out_videos_path)

    num_frames: List[int] = []
    for filename in os.listdir(raw_videos_path):
        file_path = os.path.join(raw_videos_path, filename)
        cap = cv2.VideoCapture(file_path)
        num_frames.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processor.process_video(file_path, out_videos_path)

    max_frames = 75  # int(max(num_frames))
    print(f'Max frames: {max_frames}')

    mapper = DatasetMapper(
        out_videos_path,
        f'{repo_path}/dataset',
        f'{repo_path}/WLASL_v0.3.json')
    mapper.map(max_frames, (500, 500))
