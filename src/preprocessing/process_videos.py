"""
This script is used to process raw mp4 videos (created by download_videos.sh).

It outputs hand-tracked versions using mediapipe.
"""

import json
import subprocess
from typing import List
import os


class VideoProcessor:
    """Class to apply hand-tracking to raw videos."""

    def __init__(
            self,
            mediapipe_binary_path: str,
            mediapipe_graph: str,
            mediapipe_path: str) -> None:
        """Initialize a VideoProcessor instance."""
        self.mediapipe_binary_path = mediapipe_binary_path
        self.mediapipe_graph = mediapipe_graph
        self.mediapipe_path = mediapipe_path

    def process_dir(self, src_dir: str, out_dir: str) -> None:
        """Apply hand-tracking to all videos in src_dir."""
        print(f'Processing raw videos in {src_dir}, outputting to {out_dir}')
        for filename in os.listdir(src_dir):
            file_path: str = os.path.join(src_dir, filename)
            print(f'Processing {file_path}')
            self.process_video(file_path, out_dir)

    def process_video(self, video_path: str, out_dir: str) -> None:
        """Apply hand-tracking to video and output the hand-tracked version."""
        file_name: str = os.path.basename(video_path)
        out_path: str = os.path.join(out_dir, file_name)

        # turn on logging
        os.environ['GLOG_logtostderr'] = '1'

        command: List[str] = [
            self.mediapipe_binary_path,
            f'--calculator_graph_config_file={self.mediapipe_graph}',
            f'--input_video_path={video_path}',
            f'--output_video_path={out_path}'
        ]
        print(f'Running mediapipe with command: {command}')
        subprocess.run(
            command,
            cwd=self.mediapipe_path
        )


if __name__ == "__main__":
    mediapipe_bin = '/home/rlashof/.cache/bazel/_bazel_rlashof' \
        + '/e65b822b2f7a7dfe0914d015791bc75d/execroot/mediapipe' \
        + '/bazel-out/k8-opt/bin'
    binary_path = f'{mediapipe_bin}/mediapipe/examples/desktop' \
        + '/multi_hand_tracking/multi_hand_tracking_gpu'

    mediapipe_path = '/home/rlashof/repos/mediapipe'
    graph_path = f'{mediapipe_path}/mediapipe/graphs/hand_tracking' \
        + '/multi_hand_tracking_mobile.pbtxt'
    processor = VideoProcessor(binary_path, graph_path, mediapipe_path)

    repo_path = '/home/rlashof/repos/speach-to-sign-language'
    raw_videos_path = f'{repo_path}/raw_videos_mp4'
    test_video_path = f'{raw_videos_path}/0UsjUE-TXns.mp4'
    out_videos_path = f'{repo_path}/processed_videos'

    processor.process_video(test_video_path, out_videos_path)
