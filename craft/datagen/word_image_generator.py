from multiprocessing.managers import ListProxy
from multiprocessing.synchronize import Lock
from PIL import Image, ImageDraw
from dataclasses import dataclass
import numpy as np
import cv2
import pickle
import math
from typing import Optional
from pathlib import Path

from reinlib.types.rein_color import Color
from reinlib.types.rein_color_method import ColorMethod
from reinlib.types.rein_bounding_box import BoundingBox
from reinlib.types.rein_text_layout import TextLayout
from reinlib.types.rein_size2d import Size2D
from reinlib.utility.rein_text_draw import draw_text_layout, draw_simple_text, calc_character_bboxes
from reinlib.utility.rein_image import random_pil_crop, alpha_composite

from datagen.image_generator_base import *


@dataclass
class WordImageConfig(CommonImageConfig):
    """単語画像設定
    """
    # 単語の最小、最大の長さ
    word_length_min:int
    word_length_max:int
    word_length_step:int

    # 単語の最大プロット数
    max_plots:int

    # 単語の余白サイズ
    # キャンバスの余白ではなく、単語をプロットした領域の余白です。
    word_margin_width:int
    word_margin_height:int

    def clamp_word_length(self, canvas_width:int, font_size:int) -> None:
        """単語の長さをクランプする

        Args:
            canvas_width (int): キャンバスの横幅
            font_size (int): 1文字あたりの最大横幅（おそらくはフォントサイズ）
        """
        # 余白を考慮した最大可能な最大文字数（横幅）
        word_length_max = math.floor((canvas_width - self.word_margin_width * 2) / font_size)
        if self.word_length_max > word_length_max:
            self.word_length_max = word_length_max

    @property
    def random_word_length(self) -> int:
        """ランダムな単語の長さを取得

        Returns:
            int: 単語の長さ
        """
        return random.randrange(self.word_length_min, self.word_length_max + 1, self.word_length_step)

    @property
    def word_margin(self) -> Size2D:
        """単語の余白サイズを取得

        Returns:
            Size2D: 単語の余白サイズ
        """
        return Size2D(self.word_margin_width, self.word_margin_height)


def generate_word_image(
    lock:Lock,
    background_path:str,
    canvas_size:Size2D,
    plot_margin:Size2D,
    drawer_parameters:tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...],
    characters:str,
    character_list_proxy:ListProxy,
    is_low_quality_antialias:bool,
    is_debug_enabled:bool,
    output_directory:Path,
    idx:int,
) -> None:
    # 背景画像の読込
    background_layer = Image.open(background_path).convert("RGB")
    background_layer = random_pil_crop(background_layer, canvas_size, Int2(background_layer.width, background_layer.height)//32)

    # テキスト画像
    text_layer = Image.new("RGBA", canvas_size.wh, Color.zero().rgba)
    text_drawer = ImageDraw.Draw(text_layer)

    # 低品質アンチエイリアスの作成には修飾文字を除いたテキストが必要
    if is_low_quality_antialias:
        text_alpha_layer = Image.new("L", canvas_size.wh)
        text_alpha_drawer = ImageDraw.Draw(text_alpha_layer)

    # 文字領域座標リスト
    character_bboxes:list[BoundingBox] = []

    # 文字がプロット可能な空き領域
    empty_map = np.zeros(canvas_size.hw, dtype=np.uint8)

    # デバッグが有効な場合に領域の可視化
    if is_debug_enabled:
        debug_layer = np.zeros((*canvas_size.hw, 4), dtype=np.uint8)

    # 低品質アンチエイリアス用の透明度情報
    if is_low_quality_antialias:
        text_alpha = np.zeros_like(empty_map, np.uint8)

    for word_length, custom_text, brackets, text_layout in drawer_parameters:
        # 単語の生成
        word = get_characters(lock, word_length, characters, character_list_proxy)

        # カスタムテキスト挿入後の単語
        applied_brackets_word = insert_text(word, custom_text, random.randint(0, len(word))) if custom_text is not None else word

        # 括弧類適用後の単語
        applied_brackets_word = f"{brackets[0]}{applied_brackets_word}{brackets[1]}" if brackets is not None else applied_brackets_word

        # 空き領域からテキストの描画位置を探す
        if not (info:=random_text_pos(applied_brackets_word, text_layout, canvas_size, plot_margin, empty_map)).is_enable:
            # カスタムテキストと括弧類を除いた文字列は再利用
            add_characters(lock, word, character_list_proxy)
            continue

        # 空き領域の更新
        empty_map[info.bbox_with_margin.hwslice] = 255

        if is_debug_enabled:
            cv2.rectangle(debug_layer, info.bbox_with_margin.xymin, info.bbox_with_margin.xymax, (  0,   0,   0, 128), thickness=-1)
            cv2.rectangle(debug_layer, info.bbox_decoration.xymin,  info.bbox_decoration.xymax,  (128, 128, 128, 128), thickness=-1)

        # テキスト描画
        draw_text_layout(text_drawer, info.text_pos, applied_brackets_word, text_layout, ColorMethod.ALPHA)

        # 文字領域の取得
        calc_character_bboxes(info.text_pos, applied_brackets_word, text_layout.font, text_layout.anchor, text_layout.median_bbox, character_bboxes)

        # 低品質アンチエイリアスな透明度マップの作成
        if is_low_quality_antialias:
            # 文字修飾を除いたテキスト描画
            draw_simple_text(text_alpha_drawer, info.text_pos, applied_brackets_word, text_layout.font, 255, text_layout.anchor)

            # 修飾を除いた透明度
            alpha = np.array(text_alpha_layer.crop(tuple(info.bbox_decoration)))

            # 修飾を含んだ透明度
            decoration_alpha = np.array(text_layer.crop(tuple(info.bbox_decoration)))[..., 3]

            low_quality_alpha = np.zeros_like(alpha, np.uint8)

            alpha_thresh_list = tuple([32, 64, 96, 128])
            alpha_thresh = alpha_thresh_list[random.randrange(0, len(alpha_thresh_list))]
            low_quality_alpha[alpha > alpha_thresh] = 255

            if text_layout.is_outline:
                low_quality_alpha[(decoration_alpha > 0) & (alpha <= alpha_thresh)] = text_layout.get_outline_alpha()
            elif text_layout.is_shadow:
                low_quality_alpha[(decoration_alpha > 0) & (alpha <= alpha_thresh)] = text_layout.get_shadow_alpha()
            elif text_layout.is_outline and text_layout.is_shadow:
                low_quality_alpha[(decoration_alpha > 0) & (alpha <= alpha_thresh)] = text_layout.get_outline_alpha()
            else:
                assert False, "not support low_quality_alpha."

            text_alpha[info.bbox_decoration.hwslice] = low_quality_alpha

            if False:
                cv2.imshow("", np.vstack([alpha, decoration_alpha, low_quality_alpha]))
                cv2.waitKey()
                cv2.imshow("", np.zeros_like(alpha))
                cv2.waitKey(1)

    # pillow to numpy
    background_layer = np.array(background_layer)

    # TOOLTIP
    if not is_low_quality_antialias:
        text_layer, text_alpha = np.dsplit(np.array(text_layer), (3, ))
    else:
        text_layer, _ = np.dsplit(np.array(text_layer), (3, ))
        text_alpha = text_alpha[..., np.newaxis]

    # 背景とテキストレイヤーを合成
    image_layer = alpha_composite(background_layer, text_layer, 1.0 - text_alpha / 255.0)
    image_layer = cv2.cvtColor(image_layer, cv2.COLOR_RGB2BGR)

    # デバッグが無効の場合のみ保存
    if not is_debug_enabled:
        cv2.imwrite(str(output_directory / f"image_{idx}.jpg"), image_layer)

        with open(str(output_directory / f"bboxes_{idx}.pkl"), mode="wb") as f:
            pickle.dump(character_bboxes, f, -1)
    else:
        debug_layer, debug_alpha = np.dsplit(debug_layer, (3, ))

        debug_layer = alpha_composite(background_layer, debug_layer, 1.0 - debug_alpha / 255.0)
        debug_layer = alpha_composite(debug_layer, text_layer, 1.0 - text_alpha / 255.0)

        # 文字領域
        for bbox in character_bboxes:
            cv2.rectangle(debug_layer, bbox.xymin, bbox.xymax, (0, 0, 255), 1)

        image_layer = cv2.cvtColor(image_layer, cv2.COLOR_RGB2BGR)
        empty_map = cv2.cvtColor(empty_map, cv2.COLOR_GRAY2RGB)
        text_alpha = cv2.cvtColor(text_alpha, cv2.COLOR_GRAY2RGB)

        cv2.imshow("", cv2.cvtColor(np.hstack([np.vstack([image_layer, text_alpha]), np.vstack([debug_layer, empty_map])]), cv2.COLOR_RGB2BGR))
        cv2.waitKey()
