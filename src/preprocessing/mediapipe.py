"""Wrappers and helpers to run mediapipe pipelines."""

from enum import Enum
from typing import Union, Optional, List
from pathlib import Path
import subprocess
import os


class MediapipeTarget(Enum):
    """Enum for mediapipe bazel targets."""

    MULTI_HAND_TRACKING_GPU = 1


class MediapipeRunner:
    """Wrapper class for mediapipe."""

    def __init__(self, mediapipe_path: Optional[Path] = None) -> None:
        """Initialize a MediapipeRunner instance."""
        self._temp_path: Path = Path('/tmp/speach-to-sign')
        if (mediapipe_path is None):
            self.mediapipe_path: Path = Path(f'{self._temp_path}/mediapipe')
        else:
            self.mediapipe_path = mediapipe_path
        self._is_pulled: bool = False

    def build(
            self, target: MediapipeTarget,
            gpu_build: bool) -> None:
        """Build the specified MediapipeTarget."""
        target_path: Path
        if(target == MediapipeTarget.MULTI_HAND_TRACKING_GPU):
            target_path = Path(
                'mediapipe/examples/desktop' +
                '/multi_hand_tracking:multi_hand_tracking_gpu')
        else:
            raise NotImplementedError

        base_build: str
        if(gpu_build):
            base_build = 'bazel build --verbose_failures -c opt --copt' \
                + ' -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11'
        else:
            base_build = 'bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1'

        build_cmd = f'{base_build} {target_path}'
        subprocess.run(build_cmd, cwd=self.mediapipe_path)

    def run(self, target: MediapipeTarget,
            args: Optional[List[str]] = None) -> None:
        """Run the specified target with the provided graph."""
        binary_path: str
        graph_path: str
        if(target == MediapipeTarget.MULTI_HAND_TRACKING_GPU):
            binary_path = 'bazel-bin/mediapipe/examples/desktop' \
                + '/multi_hand_tracking/multi_hand_tracking_gpu'
            graph_path = 'mediapipe/graphs/hand_tracking' \
                + '/multi_hand_tracking_desktop.pbtxt'
        else:
            raise NotImplementedError

        graph_arg = f'--calculator_graph_config_file={graph_path}'
        base_cmd: List[str] = [
            binary_path,
            graph_arg
        ]
        cmd: List[str]
        if(args is not None):
            cmd = base_cmd + args
        else:
            cmd = base_cmd

        os.environ['GLOG_logtostderr'] = '1'
        subprocess.run(cmd, cwd=self.mediapipe_path)

    def _pull(self) -> None:
        self._temp_path.mkdir(parents=True, exist_ok=True)

        if(self.mediapipe_path.exists()):
            self._is_pulled = True
            return

        git_clone = f'git clone https://github.com/google/mediapipe.git' \
            + ' {self.mediapipe_path}'
        subprocess.run(git_clone)
        self._is_pulled = True
