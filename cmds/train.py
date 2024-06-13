import argparse

from mains.train import my_app


if __name__ == "__main__":
    """モデルのトレーニング (command prompt)
    """
    parser = argparse.ArgumentParser(
        prog="モデルのトレーニング",
        description="モデルのトレーニングバッチ。",
    )
    parser.add_argument(
        "-i",
        "--dataset_directories",
        nargs="*",
        type=str,
        required=True,
        help="トレーニングに使用するデータセットのディレクトリを指定します。",
    )
    parser.add_argument(
        "-cdb",
        "--characters_database_path",
        type=str,
        help="文字情報のパスを指定します。",
    )
    parser.add_argument(
        "-max_epochs",
        type=int,
        help="トレーニング回数を指定します。",
    )
    args = parser.parse_args()

    my_app(
        args.dataset_directories,
        args.characters_database_path,
        args.max_epochs,
    )
