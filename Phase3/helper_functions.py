import os
import sys
sys.path.insert(1, '../Phase1')
from pathlib import Path
import misc


def get_images_list(folder_path):
    folder = os.path.join(Path(os.path.dirname(__file__)).parent, folder_path)
    return misc.get_images_in_directory(folder)
