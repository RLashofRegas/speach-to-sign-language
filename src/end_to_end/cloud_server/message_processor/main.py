"""Entrypoint for the message processing package for the cloud server."""
import argparse
from message_processing.processing_client import ProcessingClient
from typing import TypedDict, List
from model.predictor import Predictor
from model.model_builder import ModelBuilder


class MessageProcessingRunner:
    """Runner of the message processing client."""

    def __init__(
            self,
            broker_host: str,
            broker_port: int,
            message_channel: str,
            checkpoint_dir: str,
            checkpoint_prefix: str,
            checkpoint_number: int,
            dictionary: str) -> None:
        """Initialize the MessageProcessingRunner."""
        builder = ModelBuilder(17, (64, 64), 500, 20, 1e-4)
        predictor = Predictor(builder, dictionary, 20)
        predictor.restore_checkpoint(
            checkpoint_dir,
            checkpoint_prefix,
            checkpoint_number)
        self._processing_client = ProcessingClient(
            broker_host, broker_port, message_channel, predictor)

    def run(self) -> None:
        """Run the message processing client."""
        print('Starting message processing.')
        self._processing_client.start()


if(__name__ == "__main__"):
    arg_parser = argparse.ArgumentParser(
        description='Run the message processing pipeline.')
    arg_parser.add_argument(
        '-b', '--broker', type=str, required=True,
        help='Hostname of the message broker.')
    arg_parser.add_argument(
        '-p', '--port', type=int, required=True,
        help='Port on the broker host to connect to.')
    arg_parser.add_argument(
        '-c', '--channel', type=str, default='faces',
        help='Channel to subscribe to for incoming messages.')
    arg_parser.add_argument(
        '-h', '--checkpoint_dir', type=str, required=True,
        help='checkpoint directory.')
    arg_parser.add_argument(
        '-x', '--checkpoint_prefix', type=str, required=True,
        help='checkpoint prefix.')
    arg_parser.add_argument(
        '-n', '--checkpoint_number', type=int, required=True,
        help='checkpoint prefix.')
    arg_parser.add_argument(
        '-d', '--dictionary', type=int, required=True,
        help='Path to dictionary.')
    args = arg_parser.parse_args()
    runner = MessageProcessingRunner(
        args.broker,
        args.port,
        args.channel,
        args.checkpoint_dir,
        args.checkpoint_prefix,
        args.checkpoint_number,
        args.dictionary)
    runner.run()
