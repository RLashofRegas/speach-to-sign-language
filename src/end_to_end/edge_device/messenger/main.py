"""Entrypoint for the messaging package for the edge_device."""
import argparse
from uuid import uuid4
from messaging.messaging_client import TextMessenger
import os


class TextToSignRunner:
    """The runner of the face detection pipeline."""

    def __init__(
            self,
            output_channel: str,
            broker_host: str,
            broker_port: int,
            guarantee_level: int) -> None:
        """Initialize the runner.

        Args:
            output_channel: the channel to output messages to.
            broker_host: hostname of the message broker.
            broker_port: port of the message broker.
            video_input: input index of the video camera
                (e.g. 0 for /dev/video0)
            guarantee_level: level of guarantee for message delivery.
                0 = at most once, 1 = at least once, 2 = exactly once.
        """
        self.messenger = TextMessenger(
            output_channel,
            broker_host,
            broker_port,
            guarantee_level=guarantee_level)

    def run(self) -> None:
        """Run the text messenger pipeline."""
        self.messenger.stream_messages()


if(__name__ == "__main__"):
    devices = os.listdir('/dev')
    print(devices)
    client_id = uuid4()
    print(f'Text to sign client started for client_id={client_id}')
    arg_parser = argparse.ArgumentParser(
        description="Run the text to sign pipeline.")
    arg_parser.add_argument(
        '-c', '--channel', type=str, default=f'faces/{client_id}',
        help='Output channel to be used for publishing messages.')
    arg_parser.add_argument(
        '-b', '--broker', type=str, required=True,
        help='Hostname of the message broker.')
    arg_parser.add_argument(
        '-p', '--port', type=int, required=True,
        help='Port on the broker host to publish messages to.')
    arg_parser.add_argument(
        '-g', '--guarantee', type=int, default=0,
        help='Level of guarantee for message delivery.')
    args = arg_parser.parse_args()

    runner = TextToSignRunner(
        args.channel,
        args.broker,
        args.port,
        args.guarantee)
    runner.run()
