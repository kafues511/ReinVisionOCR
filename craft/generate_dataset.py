from mains.generate_dataset import my_app


if __name__ == "__main__":
    """データセットの生成 (vscode)
    """
    config_path_list = [
        #r"configs\msgothic001\dc4_medium.yml",
        #r"configs\msgothic001\erondo02_large.yml",
        #r"configs\msgothic001\erondo02_medium.yml",
        #r"configs\msgothic001\magicha_large.yml",
        #r"configs\msgothic001\magicha_medium.yml",
        #r"configs\msgothic001\pieces_medium.yml",
        #r"configs\msgothic001\selectoblige_large.yml",
        #r"configs\msgothic001\selectoblige_medium.yml",
        #r"configs\msgothic001\selectoblige_small.yml",

        #r"configs\mtlmr3m\haruoto_medium.yml",
        #r"configs\mtlmr3m\karumaruka_extra_medium.yml",
        #r"configs\mtlmr3m\karumaruka_large.yml",
        #r"configs\mtlmr3m\karumaruka_medium.yml",
        #r"configs\mtlmr3m\karumaruka_small.yml",
        #r"configs\mtlmr3m\riddle_joker_medium.yml",
        #r"configs\mtlmr3m\riddle_joker_small.yml",
        #r"configs\mtlmr3m\summer_pockets_medium.yml",
        #r"configs\mtlmr3m\tenshi_sz_medium.yml",

        #r"configs\SourceHanSans-Bold\otomeki_extra_large.yml",
        #r"configs\SourceHanSans-Bold\otomeki_extra_medium.yml",
        #r"configs\SourceHanSans-Bold\otomeki_extra_small.yml",
        #r"configs\SourceHanSans-Bold\otomeki_large.yml",
        #r"configs\SourceHanSans-Bold\otomeki_medium.yml",
        #r"configs\SourceHanSans-Bold\otomeki_small.yml",

        #r"configs\SourceHanSans-Medium\selectoblige_extra_large.yml",
        #r"configs\SourceHanSans-Medium\selectoblige_extra_medium.yml",
        #r"configs\SourceHanSans-Medium\selectoblige_extra_small.yml",
        #r"configs\SourceHanSans-Medium\selectoblige_large.yml",
        #r"configs\SourceHanSans-Medium\selectoblige_medium.yml",
        #r"configs\SourceHanSans-Medium\selectoblige_small.yml",
    ]

    for config_path in config_path_list:
        my_app(config_path)
