import argparse

from mains.ckpt_to_pretrained import my_app


if __name__ == "__main__":
    """チェックポイントから学習済みモデルを作成 (command prompt)
    """
    parser = argparse.ArgumentParser(
        prog="チェックポイントから学習済みモデルを作成",
        description="チェックポイントから学習済みモデルの作成バッチ。",
    )
    parser.add_argument(
        "-i",
        "--checkpoint",
        nargs="*",
        type=str,
        required=True,
        help="チェックポイントのパスかディレクトリを指定します。"
             "ディレクトリを指定した場合は最小損失のチェックポイントから学習済みモデルを作成します。",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="学習済みモデルの出力先ディレクトリを指定します。",
    )
    args = parser.parse_args()

    my_app(
        args.checkpoint,
        args.output,
        model_name="charseg",
    )
