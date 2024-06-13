from labelgen import CharacterDatabaseGenerator


__all__ = [
    "my_app",
]


def my_app(config_path:str) -> None:
    """文字情報を作成

    Args:
        config_path (str): 設定ファイルパス
    """
    CharacterDatabaseGenerator.load_from_config(config_path)
