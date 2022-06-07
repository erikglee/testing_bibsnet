th model 512, produced via training on BCP
subjects of ages 0-8 months
Greg Conan: gconan@umn.edu
Created: 2022-02-08
Updated: 2022-02-08
"""
# Import standard libraries
import argparse
from datetime import datetime 
from glob import glob
import os
import subprocess
import sys


def main():
    # Time how long the script takes and get command-line arguments from user 
    start_time = datetime.now()
    cli_args = get_cli_args()

    run_nnUNet_predict(cli_args)

    # Show user how long the pipeline took and end the pipeline here
    exit_with_time_info(start_time)


def run_nnUNet_predict(cli_args):
    """
    Run nnU-Net_predict in a subshell using subprocess
    :param cli_args: Dictionary containing all command-line input arguments
    :return: N/A
    """
    # task needs to be a string, changing type here to try and get it to work: TJH 4/26/2022
    cli_args["task"] = str(cli_args["task"])
    subprocess.check_call((cli_args["nnUNet"],
                           "-i", cli_args["input"], "-o", cli_args["output"],
                           "-t", cli_args["task"], "-m", cli_args["model"]))


def get_cli_args():
    """ 
    :return: Dictionary containing all validated command-line input arguments
    """
    script_dir = os.path.dirname(__file__)
    default_model = "3d_fullres"
    default_nnUNet_path = os.path.join(script_dir, "nnUNet_predict")
    default_task_ID = 512
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=valid_readable_dir, required=True,
        help=("Valid path to existing input directory following valid nnU-Net "
              "naming conventions (T1w files end with _0000.nii.gz and T2w "
              "end with _0001.nii.gz). There should be exactly 1 T1w file and "
              "exactly 1 T2w file in this directory.")
    )
    parser.add_argument(
        "--output", "-o", type=valid_output_dir, required=True,
    )
    parser.add_argument(
        "--nnUNet", "-n", type=valid_readable_file, default=default_nnUNet_path,
        help=("Valid path to existing executable file to run nnU-Net_predict. "
              "By default, this script will assume that nnU-Net_predict will "
              "be in the same directory as this script: {}".format(script_dir))
    )
    parser.add_argument(  # TODO Does this even need to be an argument, or will it always be the default?
        "--task", "-t", type=valid_whole_number, default=default_task_ID,
        help=("Task ID, which should be a 3-digit positive integer starting "
              "with 5 (e.g. 512).")
    )
    parser.add_argument(  # TODO Does this even need to be an argument, or will it always be the default?
        "--model", "-m", default=default_model
    )
    return validate_cli_args(vars(parser.parse_args()), parser)


def validate_cli_args(cli_args, parser):
    """
    :param cli_args: Dictionary containing all command-line input arguments
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :return: cli_args, but with all input arguments validated
    """
    # Verify that there is exactly 1 T1w file and exactly 1 T2w file in the
    # --input directory
    err_msg = ("There must be exactly 1 T{0}w file in {1} directory, but the "
               "number of T{0}w files there currently is {2}")
    t1or2_path_format = os.path.join(cli_args["input"], "*_000{}.nii.gz")
    for t1or2 in (1, 2):
        img_files = glob(t1or2_path_format.format(t1or2 - 1))
        if len(img_files) != 1:
            parser.error(err_msg.format(t1or2, cli_args["input"],
                                        len(img_files)))

    # TODO Ensure that task ID is a 3-digit number starting with 5?

    return cli_args


def valid_output_dir(path):
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, "Cannot create directory at {}",
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_readable_dir(path):
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, valid_readable_file,
                    "Cannot read directory at '{}'")


def valid_readable_file(path):
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType("r") which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, "Cannot read file at '{}'")


def valid_whole_number(to_validate):
    """
    Throw argparse exception unless to_validate is a positive integer
    :param to_validate: Object to test whether it is a positive integer
    :return: to_validate if it is a positive integer
    """
    return validate(to_validate, lambda x: int(x) >= 0, int,
                    "{} is not a positive integer")


def validate(to_validate, is_real, make_valid, err_msg, prepare=None):
    """
    Parent/base function used by different type validation functions. Raises an
    argparse.ArgumentTypeError if the input object is somehow invalid.
    :param to_validate: String to check if it represents a valid object 
    :param is_real: Function which returns true iff to_validate is real
    :param make_valid: Function which returns a fully validated object
    :param err_msg: String to show to user to tell them what is invalid
    :param prepare: Function to run before validation
    :return: to_validate, but fully validated
    """
    try:
        if prepare:
            prepare(to_validate)
        assert is_real(to_validate)
        return make_valid(to_validate)
    except (OSError, TypeError, AssertionError, ValueError,
            argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(err_msg.format(to_validate))


def exit_with_time_info(start_time, exit_code=0):
    """
    Terminate the pipeline after displaying a message showing how long it ran
    :param start_time: datetime.datetime object of when the script started
    :param exit_code: Int, exit code
    :return: N/A
    """
    print("BIBSnet for this subject took this long to run {}: {}"
          .format("successfully" if exit_code == 0 else "and then crashed",
                  datetime.now() - start_time))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
