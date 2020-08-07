"""Module exposing messaging client to read messages and save them."""
import paho.mqtt.client as mqtt
from typing import Dict
from uuid import uuid4, UUID
from model.predictor import Predictor


class ProcessingClient:
    """Client to subscribe to broker and process messages."""

    def __init__(
            self,
            broker_host: str,
            broker_port: int,
            channel: str,
            processor: Predictor) -> None:
        """Initialize the client."""
        self._host = broker_host
        self._port = broker_port
        self._channel = channel
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._processor = processor

    def _on_connect(
            self,
            _: mqtt.Client,
            __: Dict[str, str],
            ___: Dict[str, int],
            ____: int) -> None:
        print(f'Connected to message broker. Subscribing to {self._channel}')
        self._client.subscribe(f'{self._channel}/+')
        print(f'Successfully subscribed.')

    def _on_message(
            self,
            _: mqtt.Client,
            __: Dict[str, str],
            message: mqtt.MQTTMessage) -> None:
        video_name: str = str(uuid4()) + '.avi'
        print('Received message. Processing...')
        self._processor.video_for_sentence(message.payload, video_name)
        with open(video_name, 'rb') as vid:
            out_message = vid.read()
        self._client.publish(out_message, 'videos', 0)
        print('Message processed successfully.')

    def start(self) -> None:
        """Subscribe to messaging server and start processing."""
        print('Connecting to message broker...')
        self._client.connect(self._host, self._port)
        print('Connected. Starting loop...')
        self._client.loop_forever()
