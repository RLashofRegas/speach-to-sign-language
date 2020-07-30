"""Classes for processing videos."""

import json
import subprocess
from typing import List, Optional
import os
from mediapipe import MediapipeRunner, MediapipeTarget
from pathlib import Path


class HandTrackingProcessor:
    """Class to apply hand-tracking to raw videos."""

    def __init__(self, mediapipe_target: MediapipeTarget,
                 target_is_gpu: bool,
                 mediapipe_path: Optional[Path] = None) -> None:
        """Initialize a HandTrackingProcessor instance."""
        self._mediapipe_built = False
        self.mediapipe_target = mediapipe_target
        self.mediapipe_path = mediapipe_path
        self._mediapipe_runner: MediapipeRunner
        self.target_is_gpu = target_is_gpu

    def process_dir(self, src_dir: str, out_dir: str) -> None:
        """Apply hand-tracking to all videos in src_dir."""
        if(not self._mediapipe_built):
            self._build_mediapipe()

        print(f'Processing raw videos in {src_dir}, outputting to {out_dir}')
        for filename in os.listdir(src_dir):
            file_path: str = os.path.join(src_dir, filename)
            print(f'Processing {file_path}')
            self.process_video(file_path, out_dir)

    def process_video(self, video_path: str, out_dir: str) -> None:
        """Apply hand-tracking to video and output the hand-tracked version."""
        file_name: str = os.path.basename(video_path)
        out_path: str = os.path.join(out_dir, file_name)

        if(not self._mediapipe_built):
            self._build_mediapipe()

        args: List[str] = [
            f'--input_video_path={video_path}',
            f'--output_video_path={out_path}'
        ]
        self._mediapipe_runner.run(self.mediapipe_target, args)

    def _build_mediapipe(self) -> None:
        runner = MediapipeRunner(self.mediapipe_path)
        runner.build(self.mediapipe_target, gpu_build=self.target_is_gpu)
        self._mediapipe_runner = runner
        self._mediapipe_built = True
