from mains.train import my_app


if __name__ == "__main__":
    """モデルのトレーニング (vscode)
    """
    dataset_directories = [
        #r"..\resources\datasets\charseg\mtlmr3m\haruoto_medium\version_0",
        #r"..\resources\datasets\charseg\mtlmr3m\karumaruka_extra_medium\version_0",
        #r"..\resources\datasets\charseg\mtlmr3m\karumaruka_large\version_0",
        #r"..\resources\datasets\charseg\mtlmr3m\karumaruka_medium\version_0",
        #r"..\resources\datasets\charseg\mtlmr3m\karumaruka_small\version_0",
        #r"..\resources\datasets\charseg\mtlmr3m\riddle_joker_medium\version_0",
        #r"..\resources\datasets\charseg\mtlmr3m\summer_pockets_medium\version_0",
        #r"..\resources\datasets\charseg\mtlmr3m\tenshi_sz_medium\version_0",

        #r"..\resources\datasets\charseg\msgothic001\dc4_medium\version_0",
        #r"..\resources\datasets\charseg\msgothic001\erondo02_large\version_0",
        #r"..\resources\datasets\charseg\msgothic001\erondo02_medium\version_0",
        #r"..\resources\datasets\charseg\msgothic001\magicha_large\version_0",
        #r"..\resources\datasets\charseg\msgothic001\magicha_medium\version_0",
        #r"..\resources\datasets\charseg\msgothic001\pieces_medium\version_0",
        #r"..\resources\datasets\charseg\msgothic001\selectoblige_large\version_0",
        #r"..\resources\datasets\charseg\msgothic001\selectoblige_medium\version_0",
        #r"..\resources\datasets\charseg\msgothic001\selectoblige_small\version_0",

        #r"..\resources\datasets\charseg\SourceHanSans-Bold\otomeki_extra_large\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Bold\otomeki_extra_medium\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Bold\otomeki_extra_small\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Bold\otomeki_large\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Bold\otomeki_medium\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Bold\otomeki_small\version_0",

        #r"..\resources\datasets\charseg\SourceHanSans-Medium\selectoblige_extra_large\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Medium\selectoblige_extra_medium\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Medium\selectoblige_extra_small\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Medium\selectoblige_large\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Medium\selectoblige_medium\version_0",
        #r"..\resources\datasets\charseg\SourceHanSans-Medium\selectoblige_small\version_0",
    ]

    my_app(
        dataset_directories,
        10,
    )
