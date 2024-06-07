import os

from reinlib.utility.rein_files import directory_to_purepath, fast_remove


__all__ = [
    "my_app",
]


REMOVE_PATTERN_LIST = tuple([
    "**/*.jpg",
    "**/*.pkl",
])


def my_app(remove_directory:str, max_workers:int = os.cpu_count()):
    """データセットの高速削除

    Args:
        remove_directory (str): 削除するディレクトリ
        max_workers (int, optional): 削除処理に使用するスレッド数. Defaults to os.cpu_count().
    """
    if (tmp_remove_directory:=directory_to_purepath(remove_directory)) is None:
        print(f"not found directory, {str(remove_directory)}")
        return
    else:
        remove_directory = tmp_remove_directory

    fast_remove(remove_directory / "train", REMOVE_PATTERN_LIST, max_workers)
    fast_remove(remove_directory, REMOVE_PATTERN_LIST, max_workers)
