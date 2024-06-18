from mains.fast_remove import my_app


if __name__ == "__main__":
    """データセットの高速削除 (vscode)
    """
    remove_directory_list = [
        # オトメき
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_large\version_0",
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_medium\version_0",
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_small\version_0",

        # オトメき
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_large\version_0",
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_medium\version_0",
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_small\version_0",

        # Magical Charming
        #r"..\resources\datasets\craft\msgothic001\magicha_large\version_0",
        #r"..\resources\datasets\craft\msgothic001\magicha_medium\version_0",

        # こいのす
        #r"..\resources\datasets\craft\msgothic001\erondo02_large\version_0",
        #r"..\resources\datasets\craft\msgothic001\erondo02_medium\version_0",

        # D.C.4
        #r"..\resources\datasets\craft\msgothic001\dc4_medium\version_0",

        # Pieces
        #r"..\resources\datasets\craft\msgothic001\pieces_medium\version_0",

        # セレオブ
        #r"..\resources\datasets\craft\msgothic001\selectoblige_large\version_0",
        #r"..\resources\datasets\craft\msgothic001\selectoblige_medium\version_0",
        #r"..\resources\datasets\craft\msgothic001\selectoblige_small\version_0",
    ]

    for remove_directory in remove_directory_list:
        my_app(remove_directory)
