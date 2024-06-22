import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2

from reinlib.types.rein_int2 import Int2
from reinlib.types.rein_color import Color
from reinlib.types.rein_color_method import ColorMethod
from reinlib.types.rein_text_layout import TextLayout
from reinlib.types.rein_size2d import Size2D
from reinlib.utility.rein_image import random_pil_crop, alpha_composite
from reinlib.utility.rein_text_draw import draw_simple_text, draw_text_layout


__all__ = [
    "generate_character_image",
]


def generate_character_image(
    background_path:Path,
    drawer_parameters:tuple[tuple[str, Int2, TextLayout], ...],
    image_size:Size2D,
    seg_text_layout:TextLayout,
    is_debug_enabled:bool,
    output_directory:Path,
    start_idx:int,
) -> int:
    """文字画像の生成

    Args:
        background_path (Path): 背景画像パス
        drawer_parameters (tuple[tuple[str, Int2, TextLayout], ...]): 文字描画パラメータリスト
        image_size (Size2D): 画像サイズ
        is_debug_enabled (bool): デバッグの有効性
        output_directory (Path): 出力先のディレクトリ
        start_idx (int): 画像番号の開始位置

    Returns:
        int: 生成した画像枚数
    """
    # 背景画像の読込
    background_layer = Image.open(str(background_path)).convert("RGB")

    scale = 2 if max(image_size.wh) < 32 else 1

    # クリア範囲
    clear_rect = 0, 0, *image_size.wh
    clear_seg_rect = 0, 0, *(image_size * scale).wh

    # テキスト画像
    text_layer = Image.new("RGBA", image_size.wh, Color.zero().rgba)
    text_drawer = ImageDraw.Draw(text_layer)

    # セグメンテーション画像
    seg_layer = Image.new("L", (image_size * scale).wh)
    seg_drawer = ImageDraw.Draw(seg_layer)

    # 文字画像の作成
    for idx, (character, character_pos, text_layout) in enumerate(drawer_parameters):
        # テキスト＆セグメンテーション画像のクリア
        text_drawer.rectangle(clear_rect, True, 0)
        seg_drawer.rectangle(clear_seg_rect, True, 0)

        # 文字描画
        draw_text_layout(text_drawer, character_pos, character, text_layout, ColorMethod.ALPHA)

        # セグメンテーション描画
        if scale == 1:
            draw_simple_text(seg_drawer, character_pos, character, text_layout.font, 255, text_layout.anchor)
        else:
            draw_simple_text(seg_drawer, character_pos * scale, character, seg_text_layout.font, 255, seg_text_layout.anchor)

        # 背景画像をランダムに切り抜き
        new_background_layer = random_pil_crop(background_layer, image_size, Int2(*image_size) * 1.5)

        # Pillow to numpy
        new_background_layer = np.array(new_background_layer)
        new_text_layer, new_alpha_layer = np.dsplit(np.array(text_layer), (3, ))
        new_seg_layer = np.array(seg_layer)

        # 背景と文字レイヤーを合成
        new_image_layer = alpha_composite(new_background_layer, new_text_layer, 1.0 - new_alpha_layer / 255.0)
        new_image_layer = cv2.cvtColor(new_image_layer, cv2.COLOR_RGB2BGR)

        # 想定している入力サイズにリサイズ
        if (max_shape_size:=max(new_image_layer.shape)) != 32:
            new_image_layer = cv2.resize(new_image_layer, (32, 32), interpolation=(cv2.INTER_AREA if max_shape_size > 32 else cv2.INTER_LINEAR))
        if (max_shape_size:=max(new_seg_layer.shape)) != 32:
            new_seg_layer   = cv2.resize(new_seg_layer,   (32, 32), interpolation=(cv2.INTER_AREA if max_shape_size > 32 else cv2.INTER_LINEAR))

        if is_debug_enabled:
            debug_layer = np.hstack([new_image_layer, cv2.cvtColor(new_seg_layer, cv2.COLOR_GRAY2BGR)])

            cv2.imshow("", debug_layer)
            cv2.waitKey()
            cv2.imshow("", np.zeros_like(debug_layer))
            cv2.waitKey(1)
        else:
            cv2.imwrite(str(output_directory / f"image_{start_idx + idx}_{ord(character)}.jpg"), new_image_layer)
            cv2.imwrite(str(output_directory / f"label_{start_idx + idx}_{ord(character)}.png"), new_seg_layer)

    return len(drawer_parameters)
