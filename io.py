from datetime import datetime
import json
import logging
import os

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