from dataclasses import dataclass, field
import json
from typing import Self, Optional
from pathlib import Path


__all__ = [
    "CharactersDatabase",
]


@dataclass
class CharactersDatabase:
    """文字情報
    """
    # 管理番号
    version:str = "0.0"

    # ラベルから文字の変換リスト
    label_to_character_list:list[str] = field(default_factory=lambda: [])

    # 類似文字の一覧
    similar_character_table:dict[str, str] = field(default_factory=lambda: {})

    # 画像文字のリソースディレクトリの一覧
    image_character_resource_table:dict[str, str] = field(default_factory=lambda: {})

    @property
    def system_version(self) -> int:
        """文字情報のシステムバージョンを取得

        管理する情報が増えた場合にバージョンアップします。

        Returns:
            int: 文字情報のシステムバージョン
        """
        system_version, _ = self.version.split(".")
        return int(system_version)

    @system_version.setter
    def system_version(self, new_system_version:int) -> None:
        """文字情報のシステムバージョンをセット

        Args:
            new_system_version (int): 文字情報のシステムバージョン
        """
        self.version = f"{new_system_version}.{self.version}"

    @property
    def user_version(self) -> int:
        """文字情報のユーザーバージョンを取得

        ラベルや類似文字、画像文字リソースの更新などでバージョンアップします。

        Returns:
            int: 文字情報のユーザーバージョン
        """
        _, version = self.version.split(".")
        return int(version)

    @user_version.setter
    def user_version(self, new_version:int) -> None:
        """文字情報のユーザーバージョンをセット

        Args:
            new_version (int): 文字情報のユーザーバージョン
        """
        self.version = f"{self.system_version}.{new_version}"

    @staticmethod
    def suffix_exclude_dots() -> str:
        """ドット(.)を除いた拡張子を取得

        Returns:
            str: 拡張子
        """
        return "cdb"

    @staticmethod
    def suffix() -> str:
        """拡張子を取得

        Returns:
            str: 拡張子
        """
        return f".{CharactersDatabase.suffix_exclude_dots()}"

    @classmethod
    def load(cls, input_path:str | Path) -> Optional[Self]:
        """*.cdbから文字情報を読込

        Args:
            input_path (str | Path): 読み込みパス

        Returns:
            Optional[Self]: 読み込みに失敗した場合は None を返します。
        """
        if isinstance(input_path, str):
            input_path:Path = Path(input_path)
        elif not isinstance(input_path, Path):
            return None

        if input_path.suffix != CharactersDatabase.suffix():
            return None

        if not input_path.is_file():
            return None

        with open(str(input_path), mode="r", encoding="utf-8") as f:
            input_data = json.load(f)

        return cls(**input_data)

    def save(self, output_path:str | Path) -> bool:
        """文字情報を*.cdb拡張子で保存

        Args:
            output_path (str | Path): 保存先

        Returns:
            bool: 保存に失敗した場合は False を返します。
        """
        if isinstance(output_path, str):
            output_path:Path = Path(output_path)
        elif not isinstance(output_path, Path):
            return False

        if output_path.suffix != CharactersDatabase.suffix():
            output_path = output_path.with_suffix(CharactersDatabase.suffix())

        # 出力先の作成
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 出力用のJSONデータ作成
        output_data = {
            "version": self.version,
            "label_to_character_list": self.label_to_character_list,
            "similar_character_table": self.similar_character_table,
            "image_character_resource_table": self.image_character_resource_table,
        }
        output_data = json.dumps(output_data, ensure_ascii=False, indent=2)

        with open(str(output_path), mode="w", encoding="utf-8") as f:
            f.write(output_data)

        return True

    def user_version_up(self) -> None:
        """ユーザーバージョンアップ
        """
        # TODO: 明示的に呼ばないで値が変更された場合にバージョンアップ予約をTrueにする形に変更したい
        self.user_version = self.user_version + 1
