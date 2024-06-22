from mains.ckpt_to_pretrained import my_app


if __name__ == "__main__":
    """チェックポイントから学習済みモデルを作成 (vscode)
    """
    my_app(
        #r"log_logs\version_0\checkpoints",
        #r"..\resources\pretrained\mtlmr3m",
        #r"..\resources\pretrained\msgothic001",
        #r"..\resources\pretrained\SourceHanSans-Bold",
        #r"..\resources\pretrained\SourceHanSans-Medium",
        "charseg",
    )
