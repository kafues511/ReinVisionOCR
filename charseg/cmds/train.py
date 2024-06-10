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
        "-max_epochs",
        default=10,
        type=int,
        help="トレーニング回数を指定します。",
    )
    parser.add_argument(
        "-batch_size",
        default=256,
        type=int,
        help="バッチサイズを指定します。",
    )
    args = parser.parse_args()

    my_app(
        args.dataset_directories,
        args.max_epochs,
        args.batch_size,
    )
