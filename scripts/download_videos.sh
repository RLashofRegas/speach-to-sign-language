TEMP_PATH=/tmp/speach-to-sign
WLASL_PATH=$TEMP_PATH/WLASL

START_DIR=$( pwd )

if [ ! -f "${TEMP_PATH}" ]; then
	mkdir ${TEMP_PATH}
fi

if [ ! -f "${WLASL_PATH}" ]; then
	git clone https://github.com/dxli94/WLASL.git ${WLASL_PATH}
fi

cd ${WLASL_PATH}/start_kit

python video_downloader.py

bash ./scripts/swf2mp4.sh

mv ./raw_videos_mp4 ${START_DIR}

cp ./WLASL_v0.3.json ${START_DIR}

cd ${START_DIR}

rm -rf ${TEMP_PATH}