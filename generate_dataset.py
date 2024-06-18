from mains.generate_dataset import my_app


if __name__ == "__main__":
    """データセットの生成 (vscode)
    """
    config_path_list = [
        # オトメき
        #r"configs\SourceHanSans-Bold\otomeki_extra_large.yml",
        #r"configs\SourceHanSans-Bold\otomeki_extra_medium.yml",
        #r"configs\SourceHanSans-Bold\otomeki_extra_small.yml",

        # オトメき
        #r"configs\SourceHanSans-Bold\otomeki_large.yml",
        #r"configs\SourceHanSans-Bold\otomeki_medium.yml",
        #r"configs\SourceHanSans-Bold\otomeki_small.yml",

        # Magical Charming
        #r"configs\msgothic001\magicha_large.yml",
        #r"configs\msgothic001\magicha_medium.yml",

        # こいのす
        #r"configs\msgothic001\erondo02_large.yml",
        #r"configs\msgothic001\erondo02_medium.yml",

        # D.C.4
        #r"configs\msgothic001\dc4_medium.yml",

        # Pieces
        #r"configs\msgothic001\pieces_medium.yml",

        # セレオブ
        #r"configs\msgothic001\selectoblige_large.yml",
        #r"configs\msgothic001\selectoblige_medium.yml",
        #r"configs\msgothic001\selectoblige_small.yml",
    ]

    for config_path in config_path_list:
        my_app(config_path)
