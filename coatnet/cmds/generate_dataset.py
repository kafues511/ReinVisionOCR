import argparse

from mains.generate_dataset import my_app


if __name__ == "__main__":
    """データセットの生成 (command prompt)
    """
    parser = argparse.ArgumentParser(
        prog="データセットの生成",
        description="データセットの生成バッチ。",
    )
    parser.add_argument(
        "-i",
        "--config_path",
        type=str,
        required=True,
        help="設定ファイルのパスを指定します。",
    )
    args = parser.parse_args()

    my_app(args.config_path)
