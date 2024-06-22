from enum import Enum, auto
import random
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from multiprocessing.managers import ListProxy
from multiprocessing.synchronize import Lock

from reinlib.types.rein_int2 import Int2
from reinlib.types.rein_size2d import Size2D
from reinlib.types.rein_bounding_box import BoundingBox
from reinlib.types.rein_text_layout import TextLayout
from reinlib.utility.rein_text_draw import apply_font_decoration


class ImageType(Enum):
    """画像の種類
    """
    # 単語
    WORD = 0
    # 文章
    SENTENCE = auto()
    # 実際のゲーム画面を模した
    GAME = auto()

    def __str__(self) -> str:
        if self is ImageType.WORD:
            return "word"
        elif self is ImageType.SENTENCE:
            return "sentence"
        elif self is ImageType.GAME:
            return "game"
        else:
            assert False, "not support."

    def __int__(self) -> int:
        if self is ImageType.WORD:
            return ImageType.WORD.value
        elif self is ImageType.SENTENCE:
            return ImageType.SENTENCE.value
        elif self is ImageType.GAME:
            return ImageType.GAME.value
        else:
            assert False, "not support."


# 半角 to 全角
HALF2FULL = {chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)}
HALF2FULL_TABLE = str.maketrans(HALF2FULL)


# 空白、改行文字 to None
EMPTY = {" ":None, "　":None, "\n":None, "\u3000":None, "\t":None}
EMPTY_TABLE = str.maketrans(EMPTY)


@dataclass
class CommonImageConfig:
    """共通の画像設定
    """
    # 設定の有効性
    is_enable:bool

    # 画像生成数
    total:int

    # n回ごとに括弧を挿入するか
    insert_brackets_every_n_times:int

    # n回ごとにカスタムテキストを挿入するか
    insert_custom_text_every_n_times:int

    # n回ごとにランダムカラーを使用するか
    use_random_color_every_n_times:int

    @property
    def is_insert_brackets(self) -> bool:
        """括弧を挿入するか

        呼び出すたびに結果が変わります。

        Returns:
            bool: 挿入する場合は True を返します。
        """
        try:
            return next(self.insert_brackets_iter)
        except Exception as _:
            # 一定間隔だと単調なのでシャッフルしています。
            self.insert_brackets_iter = iter(random.sample([i == 0 for i in range(self.insert_brackets_every_n_times)], self.insert_brackets_every_n_times))
            return self.is_insert_brackets

    @property
    def is_insert_custom_text(self) -> bool:
        """カスタムテキストを挿入するか

        呼び出すたびに結果が変わります。

        Returns:
            bool: 挿入する場合は True を返します。
        """
        try:
            return next(self.insert_custom_text_iter)
        except Exception as _:
            # 一定間隔だと単調なのでシャッフルしています。
            self.insert_custom_text_iter = iter(random.sample([i == 0 for i in range(self.insert_custom_text_every_n_times)], self.insert_custom_text_every_n_times))
            return self.is_insert_custom_text

    @property
    def is_use_random_color(self) -> bool:
        """ランダムな文字色を使用するか

        呼び出すたびに結果が変わります。

        Returns:
            bool: 使用する場合は True を返します。
        """
        try:
            return next(self.use_random_color_iter)
        except Exception as _:
            # 一定間隔だと単調なのでシャッフルしています。
            self.use_random_color_iter = iter(random.sample([i == 0 for i in range(self.use_random_color_every_n_times)], self.use_random_color_every_n_times))
            return self.is_use_random_color

    def get_total(self) -> int:
        """有効性を考慮した画像生成数を取得

        Returns:
            int: 画像生成数
        """
        return self.total if self.is_enable else 0


def insert_text(origin:str, insert:str, index:int) -> str:
    """文字列の挿入

    Args:
        origin (str): 元の文字列
        insert (str): 挿入する文字列
        index (int): 挿入位置

    Returns:
        str: 文字列
    """
    return origin[:index] + insert + origin[index:]


@dataclass
class TextBoundingBoxInfo:
    """文字のバウンディングボックス情報
    """
    # 有効性
    is_enable:bool = False

    # 文字描画位置
    text_pos:Int2 = field(default_factory=lambda: Int2.zero())

    # 文字修飾(縁取りや影)を含んだバウンディングボックス
    bbox_decoration:BoundingBox = field(default_factory=lambda: BoundingBox.zero())

    # 余白込みのバウンディングボックス
    bbox_with_margin:BoundingBox = field(default_factory=lambda: BoundingBox.zero())


def create_text_bounding_box_info(
    text:str,
    text_pos:Int2,
    layout:TextLayout,
    canvas_size:Size2D,
) -> TextBoundingBoxInfo:
    """テキストのバウンディング情報を作成

    Args:
        text (str): テキスト
        text_pos (Int2): 文字描画位置
        layout (TextLayout): レイアウト
        canvas_size (Size2D): キャンバスサイズ

    Returns:
        TextBoundingBoxInfo: バウンディングボックス情報
    """
    info = TextBoundingBoxInfo()

    bbox = BoundingBox(*layout.font.getbbox(text, anchor=layout.anchor))
    bbox.ymin = min(layout.median_bbox.ymin, bbox.ymin)
    bbox.ymax = max(layout.median_bbox.ymax, bbox.ymax)

    bbox_decoration = BoundingBox(*bbox)
    apply_font_decoration(bbox_decoration, layout.get_outline_weight(), layout.get_shadow_weight(), layout.get_shadow_offset())

    if bbox_decoration.width >= canvas_size.width or bbox_decoration.height >= canvas_size.height:
        return info

    xmin = text_pos.x
    ymin = text_pos.y
    xmax = xmin + bbox_decoration.width
    ymax = ymin + bbox_decoration.height

    info.is_enable = True

    info.text_pos = Int2(*text_pos.xy)

    info.bbox_decoration = BoundingBox(bbox_decoration.xmin + info.text_pos.x, \
                                       bbox_decoration.ymin + info.text_pos.y, \
                                       bbox_decoration.xmax + info.text_pos.x, \
                                       bbox_decoration.ymax + info.text_pos.y)

    info.bbox_with_margin = BoundingBox(xmin + bbox_decoration.xmin, \
                                        ymin + bbox_decoration.ymin, \
                                        xmax + bbox_decoration.xmin, \
                                        ymax + bbox_decoration.ymin)

    return info


def random_text_pos(
    text:str,
    layout:TextLayout,
    canvas_size:Size2D,
    margin_size:Size2D,
    empty_map:npt.NDArray[np.uint8],
    n_trials:int = 64,
) -> TextBoundingBoxInfo:
    """文字描画位置を空き領域から探す

    Args:
        text (str): テキスト
        layout (TextLayout): レイアウト
        canvas_size (Size2D): キャンバスサイズ
        margin_size (Size2D): 余白
        empty_map (npt.NDArray[np.uint8]): 空き領域
        n_trials (int, optional): 試行回数. Defaults to 64.

    Returns:
        TextBoundingBoxInfo: バウンディングボックス情報
    """
    bbox = BoundingBox(*layout.font.getbbox(text, anchor=layout.anchor))
    bbox.ymin = min(layout.median_bbox.ymin, bbox.ymin)
    bbox.ymax = max(layout.median_bbox.ymax, bbox.ymax)

    bbox_decoration = BoundingBox(*bbox)
    apply_font_decoration(bbox_decoration, layout.get_outline_weight(), layout.get_shadow_weight(), layout.get_shadow_offset())

    if bbox_decoration.width + margin_size.width * 2 >= canvas_size.width or bbox_decoration.height + margin_size.height * 2 >= canvas_size.height:
        return TextBoundingBoxInfo()

    xrange = canvas_size.width - bbox_decoration.width - margin_size.width * 1
    yrange = canvas_size.height - bbox_decoration.height - margin_size.height * 1

    for _ in range(n_trials):
        xmin = random.randrange(abs(bbox_decoration.xmin), xrange)
        ymin = random.randrange(abs(bbox_decoration.ymin), yrange)
        xmax = xmin + bbox_decoration.width + margin_size.width * 2
        ymax = ymin + bbox_decoration.height + margin_size.height * 2

        if np.sum(empty_map[ymin + bbox_decoration.ymin:ymax + bbox_decoration.ymin,
                            xmin + bbox_decoration.xmin:xmax + bbox_decoration.xmin]) == 0:
            info = TextBoundingBoxInfo()

            info.is_enable = True

            info.text_pos = Int2(xmin + margin_size.width, \
                                 ymin + margin_size.height)

            info.bbox_decoration = BoundingBox(bbox_decoration.xmin + info.text_pos.x, \
                                               bbox_decoration.ymin + info.text_pos.y, \
                                               bbox_decoration.xmax + info.text_pos.x, \
                                               bbox_decoration.ymax + info.text_pos.y)

            info.bbox_with_margin = BoundingBox(xmin + bbox_decoration.xmin, \
                                                ymin + bbox_decoration.ymin, \
                                                xmax + bbox_decoration.xmin, \
                                                ymax + bbox_decoration.ymin)

            return info

    return TextBoundingBoxInfo()


def get_characters(lock:Lock, length:int, src:str, dst:ListProxy) -> str:
    """文字列の取得

    Args:
        lock (Lock): lock
        length (int): 文字列の長さ
        src (str): dstが空になった場合に追加する文字列
        dst (ListProxy): 文字列の取り出し

    Returns:
        str: 文字列
    """
    outs = ""

    with lock:
        for _ in range(length):
            try:
                outs += dst.pop()
            except Exception as e:
                dst.extend(random.sample(list(src), len(src)))
                outs += dst.pop()

    return outs


def add_characters(lock:Lock, src:str, dst:ListProxy) -> None:
    """文字列の追加

    Args:
        lock (Lock): lock
        src (str): 追加する文字列
        dst (ListProxy): 追加先
    """
    with lock:
        dst.extend(list(src))
