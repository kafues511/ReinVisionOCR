import argparse

from mains.generate_label import my_app


if __name__ == "__main__":
    """文字情報の生成 (command prompt)
    """
    parser = argparse.ArgumentParser(
        prog="文字情報の生成",
        description="文字情報の生成バッチ。",
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
