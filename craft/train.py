from mains.train import my_app


if __name__ == "__main__":
    """モデルのトレーニング (vscode)
    """
    dataset_directory_and_distance_ratio_table = {
        #r"..\resources\datasets\craft\msgothic001\dc4_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\erondo02_large\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\erondo02_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\magicha_large\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\magicha_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\pieces_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\selectoblige_large\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\selectoblige_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\msgothic001\selectoblige_small\version_0": 4.2,

        #r"..\resources\datasets\craft\mtlmr3m\haruoto_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\karumaruka_extra_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\karumaruka_large\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\karumaruka_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\karumaruka_small\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\riddle_joker_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\riddle_joker_small\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\summer_pockets_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\mtlmr3m\tenshi_sz_medium\version_0": 4.2,

        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_large\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_extra_small\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_large\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Bold\otomeki_small\version_0": 4.2,

        #r"..\resources\datasets\craft\SourceHanSans-Medium\selectoblige_extra_large\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Medium\selectoblige_extra_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Medium\selectoblige_extra_small\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Medium\selectoblige_large\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Medium\selectoblige_medium\version_0": 4.2,
        #r"..\resources\datasets\craft\SourceHanSans-Medium\selectoblige_small\version_0": 4.2,
    }

    dataset_directories = list(dataset_directory_and_distance_ratio_table.keys())
    dratios = list(dataset_directory_and_distance_ratio_table.values())

    my_app(
        dataset_directories,
        dratios,
    )
