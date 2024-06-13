from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Optional
import cv2

from reinlib.types.rein_int2 import Int2
from reinlib.types.rein_size2d import Size2D
from reinlib.types.rein_bounding_box import BoundingBox


__all__ = [
    "generate_character_image",
]


def generate_character_image(
    character:str,
    character_layer:npt.NDArray[np.uint8],
    padding:int,
    character_offset:Int2,
    ksize:Optional[int],
    is_debug_enabled:bool,
    output_directory:Path,
    idx:int,
) -> None:
    """文字画像の生成

    Args:
        character (str): 文字
        character_layer (npt.NDArray[np.uint8]): グレースケールな文字画像
        padding (int): 余白
        character_offset (Int2): 文字描画位置のオフセット
        ksize (Optional[int]): ガウシアンフィルタパラメータ
        is_debug_enabled (bool): デバッグの有効性
        output_directory (Path): 出力先のディレクトリ
        idx (int): 画像番号
    """
    src_size = Size2D(*character_layer.shape[::-1])
    dst_size = src_size + Size2D(padding, padding)

    src_pos = Int2(*src_size//2)
    dst_pos = Int2(*dst_size//2) + character_offset

    diff_pos = dst_pos - src_pos

    src_bbox = BoundingBox.zero()
    dst_bbox = BoundingBox.zero()

    # if: 右に移動する場合は終点を削る, else: 左に移動する場合は始点を削る
    if diff_pos.x >= 0:
        src_bbox.xmin = 0
        src_bbox.xmax = min(dst_size.width, min(src_size.width, src_size.width - diff_pos.x + padding))
    else:
        src_bbox.xmin = abs(diff_pos.x)
        src_bbox.xmax = min(src_size.width, src_size.width + src_bbox.xmin + padding)

    # if: 下に移動する場合は終点を削る, else: 上に移動する場合は始点を削る
    if diff_pos.y >= 0:
        src_bbox.ymin = 0
        src_bbox.ymax = min(dst_size.height, min(src_size.height, src_size.height - diff_pos.y + padding))
    else:
        src_bbox.ymin = abs(diff_pos.y)
        src_bbox.ymax = min(src_size.height, src_size.height + src_bbox.ymin + padding)

    dst_bbox.xmin = max(0, diff_pos.x)
    dst_bbox.ymin = max(0, diff_pos.y)
    dst_bbox.xmax = min(dst_size.width, dst_bbox.xmin + src_bbox.width)
    dst_bbox.ymax = min(dst_size.height, dst_bbox.ymin + src_bbox.height)

    image_layer = np.zeros(dst_size.hw, np.uint8)
    image_layer[dst_bbox.ymin:dst_bbox.ymax, dst_bbox.xmin:dst_bbox.xmax] = \
        character_layer[src_bbox.ymin:src_bbox.ymax, src_bbox.xmin:src_bbox.xmax]

    if ksize is not None:
        image_layer = cv2.GaussianBlur(image_layer, (ksize, ksize), 0)

    # 想定している入力サイズにリサイズ
    if max(image_layer.shape) > 32:
        image_layer = cv2.resize(image_layer, (32, 32), interpolation=cv2.INTER_AREA)
    elif max(image_layer.shape) < 32:
        image_layer = cv2.resize(image_layer, (32, 32), interpolation=cv2.INTER_LINEAR)
    else:
        image_layer = image_layer

    if is_debug_enabled:
        debug_layer = image_layer

        cv2.imshow("", debug_layer)
        cv2.waitKey()
        cv2.imshow("", np.zeros_like(debug_layer))
        cv2.waitKey(1)
    else:
        cv2.imwrite(str(output_directory / f"image_{idx}_{ord(character)}.png"), image_layer)
