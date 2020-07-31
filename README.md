# Speach to Sign Language

This repo uses the [WLASL](https://dxli94.github.io/WLASL/) dataset along with [google's mediapipe](https://google.github.io/mediapipe/) repo to train a model that can convert speach to sign language. 

# To Train

## Download Videos

You can do this part on a desktop, but for speed's sake I used an IBM Virtual Server with 8 vCPUs.

- Create Virtual Server
- Follow the instructions [here](https://docs.bazel.build/versions/master/install-ubuntu.html) to install bazel
- Install opencv:
```bash
$ apt update
$ sudo apt-get install libopencv-core-dev libopencv-highgui-dev \
                       libopencv-calib3d-dev libopencv-features2d-dev \
                       libopencv-imgproc-dev libopencv-video-dev
```
- Install pip: `sudo apt-get install python3-pip`
- Install opencv-python: `pip3 install python3-opencv`
- Install [youtube-dl](https://github.com/ytdl-org/youtube-dl#installation)
- Install ffmpeg: `apt install ffmpeg`
- Depending on system make sure python symlinks to python3 (i.e. Ubuntu): `sudo ln -s /usr/bin/python3 /usr/local/bin/python`
- Clone the repo: `git clone https://github.com/RLashofRegas/speach-to-sign-language.git`
- Run the download script. This will download and preprocess the videos using the [WLASL repo](https://github.com/dxli94/WLASL):
```bash
$ cd speach-to-sign-language/
$ bash scripts/download_videos.sh
```