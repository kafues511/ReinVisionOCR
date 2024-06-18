from mains.train import my_app


if __name__ == "__main__":
    """モデルのトレーニング (vscode)
    """
    dataset_directory_and_distance_ratio_table = {
        # オトメき
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_large\version_0" : 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_small\version_0" : 4.2,

        # オトメき
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_large\version_0" : 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_small\version_0" : 4.2,

        # Magical Charming
        #r"..\resources\datasets\craft\msgothic001\magicha_large\version_0" : 4.2,
        #r"..\resources\datasets\craft\msgothic001\magicha_medium\version_0": 4.2,

        # こいのす
        #r"..\resources\datasets\craft\msgothic001\erondo02_large\version_0" : 4.2,
        #r"..\resources\datasets\craft\msgothic001\erondo02_medium\version_0": 4.2,

        # D.C.4
        #r"..\resources\datasets\craft\msgothic001\dc4_medium\version_0" : 4.2,

        # Pieces
        #r"..\resources\datasets\craft\msgothic001\pieces_medium\version_0" : 4.2,

        # セレオブ
        #r"..\resources\datasets\craft\msgothic001\selectoblige_large\version_0" : 4.2,
        #r"..\resources\datasets\craft\msgothic001\selectoblige_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\selectoblige_small\version_0" : 4.2,
    }

    dataset_directories = list(dataset_directory_and_distance_ratio_table.keys())
    dratios = list(dataset_directory_and_distance_ratio_table.values())

    my_app(
        dataset_directories,
        dratios,
    )
