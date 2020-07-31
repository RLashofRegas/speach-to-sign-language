import json
import os
import shutil
import cv2 as cv2
import numpy as np
from typing import Tuple
import tensorflow as tf


class DatasetMapper:

    def __init__(self, src_path: str, out_path: str, index_file: str) -> None:
        self.src_path = src_path
        self.out_path = out_path
        with open(index_file) as f:
            file_index = json.load(f)

        self.file_lookup = {}
        for gloss in file_index:
            for video in gloss['instances']:
                self.file_lookup[f'{video["video_id"]}.mp4'] = {
                    'gloss': gloss['gloss'],
                    'split': video['split'],
                    'fps': video['fps']
                }

    def map(self, num_frames: int, size: Tuple[int, int]) -> None:
        if (not os.path.exists(self.out_path)):
            os.mkdir(self.out_path)

        for filename in os.listdir(self.src_path):
            file_path = os.path.join(self.src_path, filename)
            metadata = self.file_lookup[filename]

            split_path = os.path.join(self.out_path, metadata['split'])
            if (not os.path.exists(split_path)):
                os.mkdir(split_path)

            gloss_path = os.path.join(split_path, metadata['gloss'])
            if (not os.path.exists(gloss_path)):
                os.mkdir(gloss_path)

            cap = cv2.VideoCapture(file_path)
            mp4_fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_file_path = os.path.join(gloss_path, f'{filename[:-4]}.avi')
            fps = metadata['fps']
            writer = cv2.VideoWriter(
                out_file_path,
                mp4_fourcc,
                fps,
                size,
                isColor=False)

            while (True):
                ret, frame = cap.read()
                if (not ret):
                    break
                num_frames -= 1
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.write(gray_frame)
            cap.release()

            for i in range(num_frames):
                black_frame = np.zeros(size, dtype='uint8')
                writer.write(black_frame)
            writer.release()
