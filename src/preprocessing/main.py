"""Script to run preprocessing pipeline."""

from video_processors import HandTrackingProcessor
from mediapipe import MediapipeTarget
from dataset_mapper import DatasetMapper
from pathlib import Path
from typing import List, Tuple, Optional
import os
import cv2
import argparse


def run_mediapipe(
        mediapipe_path: Optional[Path],
        target: MediapipeTarget,
        videos_path: Path,
        processed_path: Path) -> int:
    use_gpu: bool
    if (target == MediapipeTarget.MULTI_HAND_TRACKING_CPU):
        use_gpu = False
    elif (target == MediapipeTarget.MULTI_HAND_TRACKING_GPU):
        use_gpu = True
    else:
        raise NotImplementedError

    processor = HandTrackingProcessor(
        target,
        use_gpu,
        mediapipe_path)

    processed_path.mkdir(parents=True, exist_ok=True)

    num_frames: List[int] = []
    for filename in os.listdir(str(videos_path)):
        file_path = os.path.join(str(videos_path), filename)
        cap = cv2.VideoCapture(file_path)
        num_frames.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processor.process_video(file_path, str(processed_path))

    max_frames = int(max(num_frames))
    print(f'Max frames: {max_frames}')
    return max_frames


def create_dataset(
        processed_path: Path,
        dataset_path: Path,
        index_file_path: Path,
        video_shape: Tuple[int, int],
        max_frames: int) -> None:
    mapper = DatasetMapper(
        str(processed_path),
        str(dataset_path),
        str(index_file_path))
    mapper.map(max_frames, video_shape)


def initialize_arguments() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description='Run the preprocessing pipeline.'
    )
    arg_parser.add_argument(
        '-m', '--mediapipe_path', type=str, default=None,
        help='Path to mediapipe repo.'
        + 'None will cause repo to be downloaded to /tmp.')
    arg_parser.add_argument(
        '-t', '--target', type=str, default='cpu',
        help='Mediapipe build/run target.')
    arg_parser.add_argument(
        '-v', '--videos', type=str, default='./videos',
        help='Path to videos to process.')
    arg_parser.add_argument(
        '-o', '--output', type=str, default='./dataset',
        help='Output dataset path.')
    arg_parser.add_argument(
        '-i', '--index', type=str, default='./WLASL_v0.3.json',
        help='Path to file index json file.')
    return arg_parser


if __name__ == "__main__":
    arg_parser = initialize_arguments()
    args = arg_parser.parse_args()

    mediapipe_path: Optional[Path] = None
    if (args.mediapipe_path is not None):
        mediapipe_path = args.mediapipe_path

    target: MediapipeTarget
    if (args.target == 'cpu'):
        target = MediapipeTarget.MULTI_HAND_TRACKING_CPU
    else:
        target = MediapipeTarget.MULTI_HAND_TRACKING_GPU

    videos_path = Path(args.videos)

    processed_path = Path('/tmp/speach-to-sign/temp_processed')

    dataset_path = Path(args.output)
    index_path = Path(args.index)

    video_shape = (500, 500)  # hard coded in mediapipe repo

    print(f'Running mediapipe for videos in {videos_path}.')
    max_frames = run_mediapipe(
        mediapipe_path,
        target,
        videos_path,
        processed_path)

    print(f'Creating dataset at {dataset_path}.')
    create_dataset(
        processed_path,
        dataset_path,
        index_path,
        video_shape,
        max_frames)
