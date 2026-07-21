import argparse
import datasets 
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from xtokenization.configs.filepaths import DATA_DIR

def parse_args(parser):
    """ Parse command line arguments for downloading samples from the Glot500c dataset.
    """
    parser.add_argument('--total_samples',  default=112000, type=int)
    parser.add_argument('--test_samples',  default=1000, type=int)
    parser.add_argument('--lang_script',type=str)
    parser.add_argument('--split',type=str, default="train")
    parser.add_argument('--file_suffix',type=str, default="samples")
    parser.add_argument('--sample_subset',type=list, default=[1000,10000,100000])
    parser.add_argument('--calibration_set_size', type=int, default=10000)
    return parser


def get_samples_from_hf(args):
    """ Download samples from the Glot500c dataset for the specified 
    language script and save the training, calibration, validation, 
    and test sets as CSV files.

    Inputs
    ----------
    args : dict
        Dictionary containing the following keys:
        - 'total_samples': Total number of samples to download from the dataset.
        - 'test_samples': Number of samples to use for the test set.
        - 'lang_script': Language script to download samples for.
        - 'split': Dataset split to use (default is "train").
        - 'file_suffix': Suffix to append to the saved CSV files (default is "samples").
        - 'sample_subset': List of training data sizes to save as separate CSV files.
        - 'calibration_set_size': Number of samples to use for the calibration set (default is 10000).
    """
    # To ensure reproducibility of the random sampling
    random_seed = 442   

    lang_data = datasets.load_dataset(
        "cis-lmu/Glot500", 
        args['lang_script'], 
        split=args["split"],
        streaming=True
    )
    # Take a random subset of the dataset 
    lang_data = lang_data.shuffle(seed=random_seed)
    lang_data = lang_data.take(args['total_samples'])

    lang_data_df = pd.DataFrame(list(lang_data))

    # Create a directory to save the CSV files
    args["save_dir"] = f"{DATA_DIR}/glot500c/"
    os.makedirs(args["save_dir"], exist_ok=True)
    
    # Split the dataset into training, calibration, validation, and test sets
    training_data = lang_data_df.iloc[:-2*args['test_samples']-args['calibration_set_size']].reset_index(drop=True)
    calibration_data = lang_data_df.iloc[-2*args['test_samples']-args['calibration_set_size']:-2*args['test_samples']].reset_index(drop=True)
    val_data = lang_data_df.iloc[-2*args['test_samples']:-args['test_samples']].reset_index(drop=True)
    test_data = lang_data_df.iloc[-args['test_samples']:].reset_index(drop=True)

    # Save the training data subsets, calibration set, validation set, and test set as CSV files
    for train_data_size in args['sample_subset']:
        lang_data_train = training_data.iloc[:train_data_size].reset_index(drop=True)
        lang_data_train.to_csv(f"""{args['save_dir']}/{args['lang_script']}_{train_data_size}{args['file_suffix']}_train.csv""", index=False)
    
    calibration_data.iloc[:int(args['calibration_set_size']/10)].to_csv(f"""{args['save_dir']}/{args['lang_script']}_{args['file_suffix']}_calibration_1000.csv""", index=False)
    calibration_data.to_csv(f"""{args['save_dir']}/{args['lang_script']}_{args['file_suffix']}_calibration_10000.csv""", index=False)
    val_data.to_csv(f"""{args['save_dir']}/{args['lang_script']}_{args['file_suffix']}_validation.csv""", index=False)
    test_data.to_csv(f"""{args['save_dir']}/{args['lang_script']}_{args['file_suffix']}_test.csv""", index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    args = vars(parser.parse_args())
    os.makedirs(args["save_dir"], exist_ok=True)
    get_samples_from_hf(args)

