"""Script to run preprocessing pipeline."""

from video_processors import HandTrackingProcessor
from mediapipe import MediapipeTarget
from pathlib import Path


if __name__ == "__main__":
    mediapipe_path = Path('/home/rlashof/repos/mediapipe')
    processor = HandTrackingProcessor(
        MediapipeTarget.MULTI_HAND_TRACKING_GPU,
        True,
        mediapipe_path)

    repo_path = '/home/rlashof/repos/speach-to-sign-language'
    raw_videos_path = f'{repo_path}/raw_videos_mp4'
    test_video_path = f'{raw_videos_path}/65225.mp4'
    out_videos_path = f'{repo_path}/processed_videos'

    processor.process_video(test_video_path, out_videos_path)
