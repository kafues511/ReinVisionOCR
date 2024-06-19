from mains.train import my_app


if __name__ == "__main__":
    """モデルのトレーニング (vscode)
    """
    dataset_directory_and_distance_ratio_table = {
        # サマポケ
        #r"..\resources\datasets\craft\mtlmr3m\summer_pockets_medium\version_0": 4.2,

        # 天使騒々
        #r"..\resources\datasets\craft\mtlmr3m\tenshi_sz_medium\version_0": 4.2,

        # Riddle Joker
        #r"..\resources\datasets\craft\mtlmr3m\riddle_joker_medium\version_0": 4.2,

        # カルマルカ (表示名が微妙にサイズ違う)
        #r"..\resources\datasets\craft\mtlmr3m\karumaruka_extra_medium\version_0": 4.2,

        # カルマルカ
        #r"..\resources\datasets\craft\mtlmr3m\karumaruka_large\version_0" : 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\karumaruka_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\karumaruka_small\version_0" : 4.2,

        # 春音
        #r"..\resources\datasets\craft\mtlmr3m\haruoto_medium\version_0": 4.2,

        # オトメき（表示名は文字修飾が違う）
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
