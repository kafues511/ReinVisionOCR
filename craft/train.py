from mains.train import my_app


if __name__ == "__main__":
    """モデルのトレーニング (vscode)
    """
    dataset_directories = [
        #r"..\resources\datasets\craft\mtlmr3m\shadow_30\version_0",
        #r"..\resources\datasets\craft\mtlmr3m\outline_shadow_30\version_0",
    ]

    dratios = [
        #4.2,
        #4.2,
    ]

    my_app(
        dataset_directories,
        dratios,
    )
