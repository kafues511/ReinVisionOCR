import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFont, ImageDraw
from typing import Optional
from pathlib import Path

from reinlib.types.rein_int2 import Int2
from reinlib.types.rein_size2d import Size2D
from reinlib.types.rein_range import Range
from reinlib.types.rein_stage_type import StageType
from reinlib.utility.rein_generate_config import GenerateConfigBase
from reinlib.utility.rein_text_draw import calc_median_bbox, draw_simple_text, calc_character_bbox, find_smallest_bounding_rectangle

from labelgen.characters_database import CharactersDatabase


__all__ = [
    "GenerateConfig",
]


class GenerateConfig(GenerateConfigBase):
    KOMIJI_HIRAGANA = "ぁぃぅぇぉっゃゅょゎゕゖ"
    KOMIJI_KATAKANA = "ァィゥェォッャュョヮヵヶ"
    KOMIJI_ENG = "ｃｏｖｗｘｚ"
    KOMOJI_LIST = KOMIJI_HIRAGANA + KOMIJI_KATAKANA + KOMIJI_ENG

    OMOJI_HIRAGANA = "あいうえおつやゆよわかけ"
    OMOJI_KATAKANA = "アイウエオツヤユヨワカケ"
    OMOJI_ENG = "ＣＯＶＷＸＺ"
    OMOJI_LIST = OMOJI_HIRAGANA + OMOJI_KATAKANA + OMOJI_ENG

    def __init__(
        self,

        font_path:str,
        font_size_list:int,

        character_database_path:str,

        padding_min:int,
        padding_max:int,
        padding_step:int,

        character_offset_step:int,

        font_size_to_apply_blur:int,

        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # フォント読込
        self.font_list = tuple([
            ImageFont.truetype(font_path, font_size)
            for font_size in font_size_list
        ])

        # 文字情報
        self.character_database = CharactersDatabase.load(character_database_path)

        # 文字情報を出力先にコピー
        if not self.is_debug_enabled:
            self.character_database.save(self.output_directory / Path(character_database_path).with_name("config").name)

        # 生成する文字の画像リスト
        self.character_images:dict[str, list[npt.NDArray[np.uint8]]] = {
            character: []
            for character in self.label_to_character_list
        }

        # 余白サイズ
        self.padding_range = Range(padding_min, padding_max + 1, padding_step)

        # 文字描画位置のオフセット幅
        self.character_offset_step = character_offset_step

        # ブラーを適用するフォントサイズ
        self.font_size_to_apply_blur = font_size_to_apply_blur

        # 文字画像を生成
        self.create_character_images()

    @property
    def label_to_character_list(self) -> tuple[str, ...]:
        """ラベルから文字変換リストを取得

        ラベル値はリストのインデックス

        Returns:
            tuple[str, ...]: ラベルから文字変換リスト
        """
        try:
            return self.__label_to_character_list
        except Exception as e:
            self.__label_to_character_list = tuple(self.character_database.label_to_character_list)
            return self.label_to_character_list

    @property
    def image_character_resource_table_keys(self) -> tuple[str, ...]:
        """画像文字をリストで取得

        Returns:
            tuple[str, ...]: 画像文字リスト
        """
        try:
            return self.__image_character_resource_table_keys
        except Exception as e:
            self.__image_character_resource_table_keys = tuple(self.character_database.image_character_resource_table.keys())
            return self.image_character_resource_table_keys

    def get_character_resource_directory(self, character:str) -> Path:
        """文字画像のディレクトリを取得

        Args:
            character (str): 文字画像に対応した文字

        Returns:
            Path: 文字画像のディレクトリ
        """
        return Path(self.character_database.image_character_resource_table[character])

    def calc_character_offsets(self, character_layer:npt.NDArray[np.uint8], padding:int, character_offset_step:int) -> tuple[Int2, ...]:
        """レイヤーサイズと余白から利用可能な文字描画位置のオフセット組み合わせを作成

        Args:
            character_layer (npt.NDArray[np.uint8]): 文字画像
            padding (int): 余白
            character_offset_step (int): 文字描画オフセット

        Returns:
            tuple[Int2, ...]: 文字描画位置のオフセットリスト
        """
        height, width = character_layer.shape

        character_smallest_bbox = find_smallest_bounding_rectangle(character_layer, 1)

        # オフセット可能なピクセル幅
        xrange = character_smallest_bbox.xmin + (width + padding - character_smallest_bbox.xmax)
        yrange = character_smallest_bbox.ymin + (height + padding - character_smallest_bbox.ymax)

        # オフセット回数
        xoffsets = xrange // character_offset_step
        yoffsets = yrange // character_offset_step

        # オフセットパターンを作成
        offsets = [
            Int2(
                -character_smallest_bbox.xmin + xoffset * character_offset_step,
                -character_smallest_bbox.ymin + yoffset * character_offset_step,
            )
            for yoffset in range(yoffsets)
            for xoffset in range(xoffsets)
        ]

        # 原点描画は必須
        if (0, 0) not in offsets:
            offsets.append(Int2.zero())

        return tuple(offsets)

    def create_character_images(self) -> None:
        """文字画像を作成
        """
        for font in self.font_list:
            # TODO: 指定された文字列だけ生成するときに中央値が崩壊する
            median_bbox = calc_median_bbox((character
                for character in self.label_to_character_list
                if character not in self.image_character_resource_table_keys
            ), font)

            # TODO: フォントサイズで画像サイズを固定化するとフォントによってははみ出るが気がする
            image_size = Size2D(font.size, font.size)
            image_layer = Image.new("L", image_size.wh)
            image_drawer = ImageDraw.Draw(image_layer)

            # 文字描画位置と塗りつぶし範囲
            fill_rect = 0, 0, *image_size.wh

            for character in self.label_to_character_list:
                if character not in self.image_character_resource_table_keys:
                    # 前回の描画結果をクリア
                    # NOTE: newより塗りつぶしのクリアの方が高速
                    image_drawer.rectangle(fill_rect, True, 0)

                    character_bbox = calc_character_bbox(Int2.zero(), character, font, "ls", median_bbox)

                    text_pos = Int2(character_bbox.xmin, -character_bbox.ymin)

                    draw_simple_text(image_drawer, text_pos, character, font, anchor="ls")

                    character_layer = np.array(image_layer)

                    self.character_images[character].append(character_layer)
                else:
                    image_path = self.get_character_resource_directory(character) / f"{font.size}.png"

                    character_layer = np.array(Image.open(str(image_path)).convert("L"))

                    self.character_images[character].append(character_layer)

    def create_train_dataset_parameters(self) -> list[tuple[str, npt.NDArray[np.uint8], int, Int2, Optional[int], bool]]:
        """データセットのパラメータを作成

        Returns:
            list[tuple[str, npt.NDArray[np.uint8], int, Int2, Optional[int], bool]]: データセットのパラメータ
        """
        return [
            (
                character,
                character_image,
                padding,
                character_offset,
                ksize,
                self.is_debug_enabled,
            )
            for character, character_images in self.character_images.items()
            for character_image in character_images
            for padding in range(self.padding_range.start, self.padding_range.stop, self.padding_range.step)
            for character_offset in self.calc_character_offsets(character_image, padding, self.character_offset_step)
            for ksize in (None, 3)
            if (ksize is None) or (max(character_image.shape) >= self.font_size_to_apply_blur)
        ]

    def create_dataset_parameters(self, stage_type:StageType) -> list[tuple[str, npt.NDArray[np.uint8], int, Int2, Optional[int], bool]]:
        """データセットのパラメータを作成

        Args:
            stage_type (StageType): ステージの種類

        Returns:
            list[tuple[str, npt.NDArray[np.uint8], int, Int2, Optional[int], bool]]: データセットのパラメータ
        """
        return super().create_dataset_parameters(stage_type)
