from mains.train import my_app


if __name__ == "__main__":
    """モデルのトレーニング (vscode)
    """
    dataset_directories = [
        #r"..\resources\datasets\charseg\mtlmr3m\shadow_30\version_0",
        #r"..\resources\datasets\charseg\mtlmr3m\outline_shadow_30\version_0",
    ]

    my_app(
        dataset_directories,
        10,
    )
