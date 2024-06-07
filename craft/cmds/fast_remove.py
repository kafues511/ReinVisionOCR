import argparse

from mains.fast_remove import my_app


if __name__ == "__main__":
    """データセットの高速削除 (command prompt)
    """
    parser = argparse.ArgumentParser(
        prog="データセットの高速削除",
        description="マルチスレッドを利用したデータセットの高速削除バッチ。"
                    "※注意: 削除挙動は『完全に削除』です。",
    )
    parser.add_argument(
        "-i",
        "--remove_directory",
        type=str,
        required=True,
        help="削除するデータセットのディレクトリを指定します。",
    )
    parser.add_argument(
        "-max_workers",
        type=int,
        help="削除に使用するワーカー（スレッド）数を指定します。",
    )
    args = parser.parse_args()

    my_app(args.remove_directory, args.max_workers)
