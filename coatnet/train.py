from mains.train import my_app


if __name__ == "__main__":
    """モデルのトレーニング (vscode)
    """
    dataset_directories = [
        #r"..\resources\datasets\coatnet\mtlmr3m\version_0",
        #r"..\resources\datasets\coatnet\msgothic001\version_0",
        #r"..\resources\datasets\coatnet\SourceHanSans-Bold\version_0",
        #r"..\resources\datasets\coatnet\SourceHanSans-Medium\version_0",
    ]

    my_app(
        dataset_directories,
    )
