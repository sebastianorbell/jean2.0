"""
Created on 09/12/2021
@author jdh
"""


def find_min_and_max_id(folder):
    paths = filter(lambda path: path.is_file(), folder.iterdir())

    ids = [int(path.stem) for path in paths]

    if ids.__len__() > 0:
        return min(ids), max(ids)
    else:
        return None, None