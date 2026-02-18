#Some Utils from abandoned repo https://github.com/M1kep/Comfy_KepListStuff

from typing import Dict, Any, List, Tuple, Optional, Iterator, Union

import PIL.Image
import numpy as np
import torch
import datetime
import re
from PIL import Image
from torch import Tensor


def validate_list_args(args: Dict[str, List[Any]]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks that if there are multiple arguments, they are all the same length or 1
    :param args:
    :return: Tuple (Status, mismatched_key_1, mismatched_key_2)
    """
    # Only have 1 arg
    if len(args) == 1:
        return True, None, None

    len_to_match = None
    matched_arg_name = None
    for arg_name, arg in args.items():
        if arg_name == 'self':
            # self is in locals()
            continue

        if len(arg) != 1:
            if len_to_match is None:
                len_to_match = len(arg)
                matched_arg_name = arg_name
            elif len(arg) != len_to_match:
                return False, arg_name, matched_arg_name

    return True, None, None

def error_if_mismatched_list_args(args: Dict[str, List[Any]]) -> None:
    is_valid, failed_key1, failed_key2 = validate_list_args(args)
    if not is_valid:
        assert failed_key1 is not None
        assert failed_key2 is not None
        raise ValueError(
            f"Mismatched list inputs received. {failed_key1}({len(args[failed_key1])}) !== {failed_key2}({len(args[failed_key2])})"
        )

def zip_with_fill(*lists: Union[List[Any], None]) -> Iterator[Tuple[Any, ...]]:
    """
    Zips lists together, but if a list has 1 element, it will be repeated for each element in the other lists.
    If a list is None, None will be used for that element.
    (Not intended for use with lists of different lengths)
    :param lists:
    :return: Iterator of tuples of length len(lists)
    """
    max_len = max(len(lst) if lst is not None else 0 for lst in lists)
    for i in range(max_len):
        yield tuple(None if lst is None else (lst[0] if len(lst) == 1 else lst[i]) for lst in lists)

def tensor2pil(image: Tensor) -> PIL.Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: PIL.Image.Image) -> Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def read_from_file(readfilename):
    try:
        with open(readfilename, 'r') as file:
            filename = file.readline().strip()
            return filename
    except FileNotFoundError:
        write_to_file(readfilename,"")
        return None

def write_to_file(writefilename, writecontent):
    try:
        with open(writefilename, 'w') as file:
            writecontent = str(writecontent)
            file.write(writecontent)
    except IOError:
        print("An error occurred while writing to " + writefilename)



# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

if __name__ == "__main__":
    # Tests
    validate_list_args({"a": [1, 2, 3], "b": [1, 2, 3]})
    validate_list_args({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    validate_list_args({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3], "d": [1, 2, 3]})
    validate_list_args({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3], "d": [1, 2, 3], "e": [1, 2, 3]})
    # Fails
    validate_list_args({"a": [1, 2, 3], "b": [1, 2, 3, 4]})

    # Tests
    print(list(zip_with_fill([1], [4, 5, 6], [8])))



def replace_date_placeholders(text):
    now = datetime.datetime.now()
    
    def replace_match(match):
        date_format = match.group(1)
        # Replace custom format with strftime format
        date_format = date_format.replace('yyyy', '%Y').replace('MM', '%m').replace('dd', '%d')
        date_format = date_format.replace('hh', '%H').replace('mm', '%M').replace('ss', '%S')
        return now.strftime(date_format)
    
    return re.sub(r'%date:([^%]+)%', replace_match, text)



# Helper to remove non-ASCII characters from strings and from tuples of strings
def _remove_non_ascii(s):
    if s is None:
        return s
    if not isinstance(s, str):
        return s
    return ''.join(c for c in s if ord(c) < 128)

def _sanitize_tuple(t):
    if not isinstance(t, tuple):
        return t
    return tuple(_remove_non_ascii(x) if isinstance(x, str) else x for x in t)

def _sanitize_filename(filename):
    """
    Removes characters that are invalid for filenames on Windows and Linux.
    Invalid characters include: < > : " / \ | ? *
    Also removes typical control characters.
    """
    if not isinstance(filename, str):
        return filename
    
    # Define invalid characters pattern
    # < > : " / \ | ? *  and control characters 0-31
    invalid_chars_pattern = r'[<>:"/\\|?*\x00-\x1f]'
    
    # Replace invalid chars with nothing or underscore - let's use underscore for visibility
    # sanitized = re.sub(invalid_chars_pattern, '', filename) # remove
    # OR
    sanitized = re.sub(invalid_chars_pattern, '', filename) # remove is safer for users' intent sometimes, but prevents collisions? 
    # Let's remove them as per likely request "remove special characters"
    
    return sanitized

