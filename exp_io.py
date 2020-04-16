from datetime import datetime
import json
import logging
import os

from keras import backend
from tensorflow import Session as tf_session
from tensorflow import ConfigProto as tf_config_proto

def create_dirs(filepath, overwrite=False, datetime_fmt='_%Y-%m-%d_%H-%M-%S'):
    """Ensures the directory path exists. Creates a sub folder with current
    datetime if it exists and overwrite is False.
    """
    # Check if dir already exists
    if not overwrite and os.path.isdir(filepath):
        logging.warning(' '.join([
            '`overwrite` is False to prevent overwriting existing directories',
            f'and a directory exists at the given filepath: `{filepath}`',
        ]))

        # NOTE beware possibility of a program writing the same file in parallel
        filepath = os.path.join(
            filepath,
            datetime.now().strftime(datetime_fmt),
        )
        os.makedirs(filepath, exist_ok=True)

        logging.warning(f'The filepath has been changed to: {filepath}')
    else:
        os.makedirs(filepath, exist_ok=True)

    return filepath


def create_filepath(
    filepath,
    overwrite=False,
    datetime_fmt='_%Y-%m-%d_%H-%M-%S',
):
    """Ensures the directories along the filepath exists. If the file exists
    and overwrite is False, then the datetime is appeneded to the filename
    while respecting the extention.

    Note
    ----
    If there is no file extension, determined via existence of a period at the
    end of the filepath with no filepath separator in the part of the path that
    follows the period, then the datetime is appeneded to the end of the file
    if it already exists.
    """
    # Check if file already exists
    if not overwrite and os.path.isfile(filepath):
        logging.warning(
            '`overwrite` is False to prevent overwriting existing files and '
            + f'there is an existing file at the given filepath: `{filepath}`'
        )

        # NOTE beware possibility of a program writing the same file in parallel
        parts = filepath.rpartition('.')
        if parts[0]:
            if parts[1] and os.path.sep in parts[1]:
                # No extension and path separator in part after period
                filepath = filepath + datetime.now().strftime(datetime_fmt)
            else:
                # Handles extension
                filepath = (
                    parts[0] + datetime.now().strftime(datetime_fmt) + parts[1]
                    + parts[2]
                )
        else:
            # handles case with no extension
            filepath = filepath + datetime.now().strftime(datetime_fmt)

        logging.warning(f'The filepath has been changed to: {filepath}')
    else:
        # ensure the directory exists
        dir_path = filepath.rpartition(os.path.sep)
        if dir_path[0]:
            os.makedirs(dir_path[0], exist_ok=True)

    return filepath


def add_logging_args(parser, default_log_level='WARNING'):
    parser.add_argument(
        '--log_level',
        default=default_log_level,
        help='The log level to be logged.',
        choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    )
    parser.add_argument(
        '--log_file',
        default=None,
        type=str,
        help='The log file to be written to.',
    )


def add_hardware_args(parser):
    """Adds the arguments detailing the hardware to be used."""
    # TODO consider packaging as a dict/NestedNamespace
    # TODO consider a boolean or something to indicate when to pass a
    # tensorflow session or to use it as default

    parser.add_argument(
        '--cpu',
        default=1,
        type=int,
        help='The number of available CPUs.',
    )
    parser.add_argument(
        '--cpu_cores',
        default=1,
        type=int,
        help='The number of available cores per CPUs.',
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )
    parser.add_argument(
        '--which_gpu',
        default=None,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )


def get_tf_config(cpu_cores=1, cpus=1, gpus=0, allow_soft_placement=True):
    return tf_config_proto(
        intra_op_parallelism_threads=cpu_cores,
        inter_op_parallelism_threads=cpu_cores,
        allow_soft_placement=allow_soft_placement,
        device_count={
            'CPU': cpus,
            'GPU': gpus,
        } if gpus >= 0 else {'CPU': cpus},
    )


def set_hardware(args):
    # Set the Hardware
    backend.set_session(tf_session(config=get_tf_config(
        args.cpu_cores,
        args.cpu,
        args.gpu,
    )))

def set_logging(args):
    # Set logging configuration
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    if args.log_file is not None:
        dir_part = args.log_file.rpartition(os.path.sep)[0]
        os.makedirs(dir_part, exist_ok=True)
        # TODO add optional non-overwrite of existing logs using exp_io
        logging.basicConfig(filename=args.log_file, level=numeric_level)
    else:
        logging.basicConfig(level=numeric_level)
