from datagen import DatasetGenerator


__all__ = [
    "my_app",
]


def my_app(config_path:str) -> None:
    """データセットの生成

    Args:
        config_path (str): 設定ファイルパス
    """
    DatasetGenerator(config_path).generate()
