import cv2
import paho.mqtt.client as mqtt
from uuid import uuid4, UUID


def on_connect(client, __, ___, ____):
    client.subscribe('videos')


def on_message(client, __, message):
    video_name: str = str(uuid4()) + '.avi'
    with open(video_name, 'wb') as vid:
        vid.write(message.payload)
    cap = cv2.VideoCapture(video_name)
    while(cap.isOpened()):
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    client = mqtt.Client()
    client.on_message = on_message
    client.on_connect = on_connect
    client.connect('message_broker', 1883)
    client.loop_forever()
