from pathlib import Path
import argparse
import shutil

arg_parser = argparse.ArgumentParser(
        description='Run the preprocessing pipeline.'
    )
arg_parser.add_argument(
    '-e', '--existing_dictionary', type=str, default='dictionary.txt',
    help='Path to existing dictionary to break up.')
arg_parser.add_argument(
    '-s', '--dataset_dir', type=str, default='dataset',
    help='Path to existing dataset to break up.')
arg_parser.add_argument(
    '-n', '--num_words', type=int, required=True,
    help='Number of words per dictionary.')

args = arg_parser.parse_args()

dict_path = Path(args.existing_dictionary)
dict_parent = dict_path.parent
dict_base_name = dict_path.stem
dict_extension = dict_path.suffix

dataset_path = Path(args.dataset_dir)
dataset_base_name = dataset_path.stem
dataset_parent = dataset_path.parent


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

words = []
with open(str(dict_path), 'r') as d:
    for batch_index, word_batch in enumerate(batch(d.readlines(), n=args.num_words)):
        new_dict_path = dict_parent / f'{dict_base_name}_{batch_index}{dict_extension}'
        new_dataset_path = dataset_parent / f'{dataset_base_name}_{batch_index}'
        with open(str(new_dict_path), 'w') as new_dict:
            for word in word_batch:
                new_dict.write(word)
                for folder in dataset_path.glob(f'*/{word.strip()}'):
                    tree = Path(*folder.parts[-2:])
                    new_path = new_dataset_path / tree
                    shutil.copytree(folder, new_path)

        

