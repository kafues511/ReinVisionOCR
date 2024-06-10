import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from typing import Optional, Iterator
from pathlib import Path
import random
from copy import deepcopy

from reinlib.types.rein_int2 import Int2
from reinlib.types.rein_color import Color
from reinlib.types.rein_stage_type import StageType
from reinlib.types.rein_text_layout import TextLayout
from reinlib.types.rein_size2d import Size2D
from reinlib.types.rein_range import Range
from reinlib.types.rein_color import HSV
from reinlib.utility.rein_files import get_paths_in_directories
from reinlib.utility.rein_text_draw import calc_median_bbox, calc_character_bbox, draw_simple_text, find_smallest_bounding_rectangle
from reinlib.utility.rein_generate_config import GenerateConfigBase


__all__ = [
    "GenerateConfig",
]


class GenerateConfig(GenerateConfigBase):
    # 背景画像の対応拡張子
    IMAGE_SUFFIXES = tuple([".jpg"])

    def __init__(
        self,

        font_path:str,
        font_size:int,

        background_image_directories:tuple[str, ...],

        text_color:tuple[int, int, int],
        text_alpha_min:int,
        text_alpha_max:int,
        text_alpha_step:int,

        is_outline:bool,
        outline_weight:int,
        outline_color:tuple[int, int, int],
        outline_alpha_min:int,
        outline_alpha_max:int,
        outline_alpha_step:int,

        is_shadow:bool,
        shadow_weight:int,
        shadow_offset:tuple[int, int],
        shadow_color:tuple[int, int, int],
        shadow_alpha_min:int,
        shadow_alpha_max:int,
        shadow_alpha_step:int,

        is_apply_same_alpha:bool,

        hue_min:int,
        hue_max:int,
        hue_step:int,

        saturation_min:int,
        saturation_max:int,
        saturation_step:int,

        padding_min:int,
        padding_max:int,
        padding_step:int,

        character_offset_step:int,

        characters_per_image:int,

        character_parameters:tuple[dict[str, str | int]],

        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # フォント読込と対応文字をUnicodeで取得
        self.font = ImageFont.truetype(font_path, font_size)
        self.cmap:tuple[int, ...] = tuple(TTFont(font_path).getBestCmap())

        # 背景画像パスの読込
        self.background_image_paths = get_paths_in_directories(background_image_directories, self.IMAGE_SUFFIXES)

        # 文字色
        self.text_color = text_color
        self.text_alpha_range = Range(text_alpha_min, text_alpha_max + 1, text_alpha_step)

        # 縁取り
        self.is_outline = is_outline
        self.outline_weight = outline_weight
        self.outline_color = outline_color
        self.outline_alpha_range = Range(outline_alpha_min, outline_alpha_max + 1, outline_alpha_step)

        # 影付き
        self.is_shadow = is_shadow
        self.shadow_weight = shadow_weight
        self.shadow_offset = Int2(*shadow_offset)
        self.shadow_color = shadow_color
        self.shadow_alpha_range = Range(shadow_alpha_min, shadow_alpha_max + 1, shadow_alpha_step)

        # 縁取りと影付きの透明度を共通にするか
        self.is_apply_same_alpha = is_apply_same_alpha

        # 色相と彩度の範囲
        self.hue_range = Range(hue_min, hue_max + 1, hue_step)
        self.saturation_range = Range(saturation_min, saturation_max + 1, saturation_step)

        # 文字変換表の作成
        character_translation_table:dict[int, Optional[str]] = {}
        character_translation_table |= str.maketrans({" ":None, "　":None, "\n":None, "\u3000":None, "\t":None})      # 空白文字の削除
        character_translation_table |= str.maketrans({chr(0x0021 + code): chr(0xFF01 + code) for code in range(94)})  # 半角から全角の変換

        # 生成文字リスト
        characters:list[str] = []

        # 生成文字の読込
        for character_parameter in character_parameters:
            self.register_characters(*character_parameter.values(), character_translation_table, characters)

        # list to str
        self.characters = "".join(characters)

        # 重複なしの生成文字一覧を作成
        non_duplicate_characters = set(self.characters)

        # 生成文字リストから文字領域の中央値のバウンディングボックスを算出
        # NOTE: 文字描画位置の算出に使用します。
        self.median_bbox = calc_median_bbox(non_duplicate_characters, self.font)

        # 余白
        self.padding_range = Range(padding_min, padding_max + 1, padding_step)
        self.padding_num = len([True for _ in range(*self.padding_range)])

        # 文字描画位置のオフセット幅
        self.character_offset_step = character_offset_step

        # 背景画像あたりに生成する文字数
        self.characters_per_image = characters_per_image

        # 余白ごとの文字描画位置リスト
        self.character_pos_table:dict[str, dict[int, tuple[Int2, ...]]] = {
            character: {padding: None for padding in range(*self.padding_range)}
            for character in non_duplicate_characters
        }

        # 余白ごとのランダムな文字描画を取得するイテレータ
        self.character_pos_iter:dict[str, dict[int, Iterator[str]]] = deepcopy(self.character_pos_table)

        # セグメンテーション用のテキストレイアウト
        self.seg_text_layout = TextLayout(font=ImageFont.truetype(font_path, font_size * 2), anchor="ls")

        # 余白ごとの文字描画位置リストの作成
        self.create_character_pos_table()

    def register_characters(
        self,
        characters_or_path:str,
        repeat:int,
        character_translation_table:dict[int, Optional[str]],
        out_characters:list[str],
    ) -> None:
        """文字列の読込

        Args:
            characters_or_path (str): 文字列もしくはテキストファイルパス
            repeat (int): 文字列の繰り返し数
            character_translation_table (dict[int, Optional[str]]): 文字列の変換表
            out_characters (list[str]): 読み込んだ文字列の格納先
        """
        # read text file.
        if characters_or_path.rfind(".txt") != -1:
            with open(characters_or_path, mode="r", encoding="utf-8") as f:
                characters_or_path = f.read()

        # half to full & remove empty
        characters_or_path = characters_or_path.translate(character_translation_table)

        # 未対応文字を除外
        characters_or_path = tuple([char for char in characters_or_path if ord(char) in self.cmap])

        # apply repeat.
        out_characters.extend(characters_or_path * repeat)

    def calc_character_positions(
        self,
        character_pos:Int2,
        character_layer:npt.NDArray[np.uint8],
        padding:int,
    ) -> tuple[Int2, ...]:
        """レイヤーサイズと余白から利用可能な文字描画位置の組み合わせを作成

        Args:
            character_pos (Int2): 文字描画位置 (原点)
            character_layer (npt.NDArray[np.uint8]): 文字画像
            padding (int): 余白

        Returns:
            tuple[Int2, ...]: オフセットを考慮した文字描画位置リスト
        """
        height, width = character_layer.shape

        character_smallest_bbox = find_smallest_bounding_rectangle(character_layer, 1)

        # オフセット可能なピクセル幅
        xrange = character_smallest_bbox.xmin + (width + padding - character_smallest_bbox.xmax)
        yrange = character_smallest_bbox.ymin + (height + padding - character_smallest_bbox.ymax)

        # オフセット回数
        xoffsets = xrange // self.character_offset_step
        yoffsets = yrange // self.character_offset_step

        # オフセットを考慮した文字描画位置を作成
        offsets = [
            Int2(
                character_pos.x + -character_smallest_bbox.xmin + xoffset * self.character_offset_step,
                character_pos.y + -character_smallest_bbox.ymin + yoffset * self.character_offset_step,
            )
            for yoffset in range(yoffsets)
            for xoffset in range(xoffsets)
        ]

        # 原点描画は必須
        if character_pos not in offsets:
            offsets.append(character_pos)

        return tuple(offsets)

    def create_character_pos_table(self) -> None:
        """余白ごとの文字描画位置リストの作成
        """
        # TODO: フォントサイズで画像サイズを固定化するとフォントによってははみ出るが気がする
        image_size = Size2D(self.font.size, self.font.size)
        image_layer = Image.new("L", image_size.wh)
        image_drawer = ImageDraw.Draw(image_layer)

        # 文字描画位置と塗りつぶし範囲
        fill_rect = 0, 0, *image_size.wh

        for character in set(self.characters):
            # 前回の描画結果をクリア
            # NOTE: newより塗りつぶしのクリアの方が高速
            image_drawer.rectangle(fill_rect, True, 0)

            character_bbox = calc_character_bbox(Int2.zero(), character, self.font, "ls", self.median_bbox)

            character_pos = Int2(character_bbox.xmin, -character_bbox.ymin)

            draw_simple_text(image_drawer, character_pos, character, self.font, anchor="ls")

            character_layer = np.array(image_layer)

            for padding in range(*self.padding_range):
                self.character_pos_table[character][padding] = self.calc_character_positions(character_pos, character_layer, padding)

    @property
    def character(self) -> str:
        """文字を取得

        Returns:
            str: 文字
        """
        try:
            return next(self.character_iter)
        except Exception as _:
            self.character_iter = iter(random.sample(self.characters, len(self.characters)))
            return self.character

    @property
    def random_text_color(self) -> tuple[int, int, int]:
        """ランダムな文字色を取得

        Returns:
            tuple[int, int, int]: ランダムな文字色 (RGB)
        """
        return Color.from_hsv(HSV(self.hue_range(), self.saturation_range(), 100)).rgb

    def get_character_pos(self, character:str, padding:int) -> Int2:
        """文字描画位置を取得

        Args:
            character (str): 文字
            padding (int): 余白

        Returns:
            Int2: 文字描画位置
        """
        try:
            return next(self.character_pos_iter[character][padding])
        except Exception as _:
            self.character_pos_iter[character][padding] = iter(random.sample(self.character_pos_table[character][padding], len(self.character_pos_table[character][padding])))
            return self.get_character_pos(character, padding)

    def create_text_layout(self, is_random_text_color:bool = False) -> TextLayout:
        """テキストレイアウトの作成

        Args:
            is_random_text_color (bool, optional): ランダムな文字色を使用するか. Defaults to False.

        Returns:
            TextLayout: テキストレイアウト
        """
        # 文字色
        if is_random_text_color:
            text_color = self.random_text_color
        else:
            text_color = self.text_color

        # 縁取りと影付きの透明度
        if self.is_apply_same_alpha:
            shadow_alpha = self.shadow_alpha_range()
            outline_alpha = shadow_alpha
        else:
            outline_alpha = self.outline_alpha_range()
            shadow_alpha = self.shadow_alpha_range()

        return TextLayout(
            self.font,
            Color(*text_color, self.text_alpha_range()),
            self.is_outline,
            Color(*self.outline_color, outline_alpha),
            self.outline_weight,
            self.is_shadow,
            Color(*self.shadow_color, shadow_alpha),
            self.shadow_weight,
            self.shadow_offset,
            anchor="ls",
            median_bbox=self.median_bbox,
        )

    @property
    def characters_per_executor(self) -> int:
        """Executorあたりに生成する文字数を取得

        Returns:
            int: Executorあたりに生成する文字数
        """
        return int(self.characters_per_image / self.padding_num)

    @property
    def total(self) -> int:
        """生成文字画像の総数を取得

        Returns:
            int: 生成文字画像の総数
        """
        return int(len(self.background_image_paths) * self.characters_per_executor * self.padding_num)

    def create_drawer_parameters(self, padding:int) -> tuple[tuple[str, Int2, TextLayout], ...]:
        """文字描画パラメータリストの作成

        Args:
            padding (int): 余白サイズ

        Returns:
            tuple[tuple[str, Int2, TextLayout], ...]: 文字描画パラメータリスト
        """
        parameters:list[tuple[str, Int2, TextLayout]] = []

        for _ in range(self.characters_per_executor):
            character = self.character
            character_offset = self.get_character_pos(character, padding)
            text_layout = self.create_text_layout()
            parameters.append((character, character_offset, text_layout))

        return tuple(parameters)

    def create_train_dataset_parameters(self) -> list[tuple[Path, tuple[tuple[str, Int2, TextLayout], ...], bool]]:
        """データセットのパラメータを作成

        Returns:
            list[tuple[Path, tuple[tuple[str, Int2, TextLayout], ...], bool]]: データセットのパラメータ
        """
        return [
            (
                background_image_path,
                self.create_drawer_parameters(padding),
                Size2D.fill(self.font.size + padding),
                self.seg_text_layout,
                self.is_debug_enabled,
            )
            for background_image_path in self.background_image_paths
            for padding in range(*self.padding_range)
        ]

    def create_dataset_parameters(self, stage_type:StageType) -> list[tuple[Path, tuple[tuple[str, Int2, TextLayout], ...], bool]]:
        """データセットのパラメータを作成

        Args:
            stage_type (StageType): ステージの種類

        Returns:
            list[tuple[Path, tuple[tuple[str, Int2, TextLayout], ...], bool]]: データセットのパラメータ
        """
        return super().create_dataset_parameters(stage_type)
