import os
from typing import Optional
from fontTools.ttLib import TTFont
from pathlib import Path

from labelgen.characters_database import CharactersDatabase
from reinlib.utility.rein_yml import YMLLoader


__all__ = [
    "CharacterDatabaseGenerator",
]


class CharacterDatabaseGenerator(YMLLoader):
    def __init__(
        self,
        output_directory:str,
        is_new_create:bool,
        font_path:str,
        ignore_characters:tuple[str, ...],
        similar_character_groups:tuple[dict[str, str | tuple[str, ...]], ...],
        character_parameters:tuple[str, ...],
    ) -> None:
        """コンストラクタ

        Args:
            output_directory (str): 出力先ディレクトリ
            is_new_create (bool): 文字情報を新規作成するか
            font_path (str): ラベル作成に使用するフォントのパス
            ignore_characters (tuple[str, ...]): ラベルから除外する文字
            similar_character_groups (tuple[dict[str, str  |  tuple[str, ...]], ...]): 類似文字
            character_parameters (tuple[str, ...]): 生成文字パラメータリスト
        """
        # 文字情報の入出力パス
        character_database_path = self.create_character_database_path(output_directory, font_path)

        # if:   文字情報の読込
        # else: 文字情報の新規作成
        if not is_new_create and (cdb:=CharactersDatabase.load(character_database_path)) is not None:
            self.character_database = cdb
            self.character_database.user_version_up()
        else:
            self.character_database = CharactersDatabase()

        # フォントが対応している文字コードを取得
        self.cmap:tuple[int, ...] = tuple(TTFont(font_path).getBestCmap())

        # 文字変換表の作成
        self.character_translation_table:dict[int, Optional[str]] = {}
        self.character_translation_table |= str.maketrans({character: None for character in ignore_characters})            # 除外文字の削除
        self.character_translation_table |= str.maketrans({" ":None, "　":None, "\n":None, "\u3000":None, "\t":None})      # 空白文字の削除
        self.character_translation_table |= str.maketrans({chr(0x0021 + code): chr(0xFF01 + code) for code in range(94)})  # 半角から全角の変換

        # 類似文字グループの子文字列を親文字に変換するテーブル
        self.similar_character_translation_table = str.maketrans({
            similar_character_child: similar_character_group["parent"]
            for similar_character_group in similar_character_groups
            for similar_character_child in similar_character_group["childs"]
        })

        # ラベル振り分けの開始
        for character_parameter in character_parameters:
            if self.register_from_text_file(character_parameter):
                pass
            elif self.register_from_image_character(character_parameter):
                pass
            elif self.register_from_characters(character_parameter):
                pass
            else:
                assert False, "not support 'character_parameter' format."

        # 全角スペースを登録
        # 有効ピクセルが存在しない不正な領域が入力された場合に全角スペースとして識字することを期待しています。
        if not self.register_from_characters("　", is_force=True):
            assert False, "failed to register double-byte white space."

        # 類似文字表
        self.similar_character_table = {
            chr(child): parent
            for child, parent in self.similar_character_translation_table.items()
        }

        # 文字情報の保存
        self.character_database.save(character_database_path)

    @staticmethod
    def create_character_database_path(output_directory:str, font_path:str) -> Path:
        """文字情報の読込/保存先パスを作成

        Args:
            output_directory (str): 出力先（文字情報の読込/保存先）
            font_path (str): フォントパス（フォント名を使用する）

        Returns:
            Path: 作成されたパス
        """
        # TODO: ないだろうが font_path に拡張子がない場合の分岐作成
        return Path(output_directory) / f"{Path(font_path).with_suffix(CharactersDatabase.suffix()).name}"

    @property
    def label_to_character_list(self) -> list[str]:
        """ラベルから文字変換リストを取得

        ラベル値はリストのインデックス

        Returns:
            list[str]: ラベルから文字変換リスト
        """
        return self.character_database.label_to_character_list

    @label_to_character_list.setter
    def label_to_character_list(self, new_value:list[str]) -> None:
        """ラベルから文字変換リストをセット

        Args:
            new_value (list[str]): ラベルから文字変換リスト
        """
        self.character_database.label_to_character_list = new_value

    @property
    def similar_character_table(self) -> dict[str, str]:
        """類似文字の一覧を取得

        Returns:
            dict[str, str]: 類似文字の一覧
        """
        return self.character_database.similar_character_table

    @similar_character_table.setter
    def similar_character_table(self, new_value:dict[str, str]) -> None:
        """類似文字の一覧をセット

        Args:
            new_value (dict[str, str]): 類似文字の一覧
        """
        self.character_database.similar_character_table = new_value

    @property
    def image_character_resource_table(self) -> dict[str, str]:
        """画像文字のリソースディレクトリの一覧を取得

        Returns:
            dict[str, str]: 画像文字のリソースディレクトリの一覧
        """
        return self.character_database.image_character_resource_table

    @image_character_resource_table.setter
    def image_character_resource_table(self, new_value:dict[str, str]) -> None:
        """画像文字のリソースディレクトリの一覧をセット

        Args:
            new_value (dict[str, str]): 画像文字のリソースディレクトリの一覧
        """
        self.character_database.image_character_resource_table = new_value

    def register_from_text_file(self, path:str) -> bool:
        """テキストファイルから文字列の登録

        Args:
            path (str): テキストファイルパス

        Returns:
            bool: 登録結果、テキストファイル以外が指定された場合は False を返します。
        """
        if not path.rfind(".txt") or not os.path.isfile(path):
            return False

        with open(path, mode="r", encoding="utf-8") as f:
            characters = f.read()

        return self.register_from_characters(characters)

    def register_from_image_character(self, directory:str) -> bool:
        """画像文字から文字の登録

        Args:
            directory (str): 画像文字が配置されているディレクトリ

        Returns:
            bool: 画像文字でない場合は False を返します。
        """
        if not os.path.isdir(directory) or \
           not os.path.isfile(path:=(os.path.join(directory, "code.txt"))):
            return False

        with open(path, mode="r", encoding="utf-8") as f:
            character = f.read()

        # 画像文字は1つのディレクトリに付き1文字
        if len(character) != 1:
            return False

        # 画像の文字のディレクトリを記録
        self.image_character_resource_table |= {character: os.path.abspath(directory)}

        return self.register_from_characters(character, is_force=True)

    def register_from_characters(self, characters:str, is_force:bool = False) -> bool:
        """文字列を登録

        Args:
            characters (str): 登録する文字列
            is_force (bool, optional): 有効にした場合は 文字変換 と フォントに含まれる文字列か判定 を無視して追加します. Defaults to False.

        Returns:
            bool: 文字列の登録結果
        """
        if not is_force:
            characters = characters.translate(self.character_translation_table)
            characters = characters.translate(self.similar_character_translation_table)

        for character in characters:
            if (is_force or ord(character) in self.cmap) and character not in self.label_to_character_list:
                self.label_to_character_list += character

        return True
