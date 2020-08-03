from pathlib import Path
import argparse

arg_parser = argparse.ArgumentParser(
        description='Run the preprocessing pipeline.'
    )
arg_parser.add_argument(
    '-d', '--dictionary_path', type=str, default='dictionary.txt',
    help='Path to where dictionary will be created.')
arg_parser.add_argument(
    '-s', '--dataset_path', type=str, default='dataset',
    help='Path of where to look for the dataset.')

args = arg_parser.parse_args()

dict_path = Path(args.dictionary_path)
dataset_path = Path(args.dataset_path)

with open(str(dict_path), 'w') as dict_file:
    for file_path in dataset_path.glob('*/*'):
        word = file_path.parts[-1]
        dict_file.write(f'{word}\n')
