from PIL import Image, ImageDraw
from dataclasses import dataclass
import numpy as np
import cv2
import pickle
from pathlib import Path

from reinlib.types.rein_int2 import Int2
from reinlib.types.rein_size2d import Size2D
from reinlib.types.rein_range import Range
from reinlib.types.rein_hsv import HSV
from reinlib.types.rein_color import Color
from reinlib.types.rein_color_method import ColorMethod
from reinlib.types.rein_bounding_box import BoundingBox
from reinlib.types.rein_text_layout import TextLayout
from reinlib.utility.rein_files import get_paths_in_directories
from reinlib.utility.rein_image import random_pil_crop, alpha_composite, create_gradient_alpha
from reinlib.utility.rein_text_draw import draw_simple_text, draw_text_layout, calc_character_bboxes, apply_font_decoration

from datagen.image_generator_base import *


@dataclass
class GameImageConfig(CommonImageConfig):
    # 対応画像の拡張子
    IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

    """ゲーム画像設定
    """
    # テキストの長さの範囲
    text_length_min:int
    text_length_max:int
    text_length_step:int

    # 行間サイズ
    spacing:int

    # 背景の横幅の範囲
    background_width_min:int
    background_width_max:int
    background_width_step:int

    # 背景の縦幅の範囲
    background_height_min:int
    background_height_max:int
    background_height_step:int

    # テキストボックスの色相の範囲
    text_box_hue_min:int
    text_box_hue_max:int
    text_box_hue_step:int

    # テキストボックスの彩度の範囲
    text_box_saturation_min:int
    text_box_saturation_max:int
    text_box_saturation_step:int

    # テキストボックス画像のディレクトリ
    text_box_image_directories:tuple[str, ...]

    # n回ごとにテキストボックス画像を使用するか
    use_text_box_image_every_n_times: int

    # テキストボックスの上部の透明度の範囲
    # text_box_top_alpha ~ text_box_bottom_alpha でグラデーションします。
    text_box_top_alpha_min:int
    text_box_top_alpha_max:int
    text_box_top_alpha_step:int

    # テキストボックスの下部の透明度の範囲
    # text_box_top_alpha ~ text_box_bottom_alpha でグラデーションします。
    text_box_bottom_alpha_min:int
    text_box_bottom_alpha_max:int
    text_box_bottom_alpha_step:int

    # キャンバス色
    canvas_color:tuple[int, int, int] | Color

    def __post_init__(self) -> None:
        # ディレクトリに含まれる画像パスを取得
        self.text_box_image_paths:tuple[Path, ...] = tuple(get_paths_in_directories(self.text_box_image_directories, self.IMAGE_SUFFIXES))

        # tuple[int, int, int] から Colroに変換
        self.canvas_color:Color = Color(*self.canvas_color, 255)

    @property
    def random_text_length(self) -> int:
        """ランダムなテキストの長さを取得

        Returns:
            int: テキストの長さ
        """
        return random.randrange(self.text_length_min, self.text_length_max + 1, self.text_length_step)

    @property
    def background_width_range(self) -> Range:
        """背景の横幅の範囲を取得

        Returns:
            Range: start, stop, step が格納された Range を返します。
        """
        return Range(self.background_width_min, self.background_width_max + 1, self.background_width_step)

    @property
    def background_height_range(self) -> Range:
        """背景の縦幅の範囲を取得

        Returns:
            Range: start, stop, step が格納された Range を返します。
        """
        return Range(self.background_height_min, self.background_height_max + 1, self.background_height_step)

    @property
    def random_text_box_image_path(self) -> str:
        """ランダムなテキストボックスの画像パスを取得

        Returns:
            str: テキストボックスの画像パス
        """
        try:
            return str(next(self.text_box_image_path_iter))
        except Exception as _:
            self.text_box_image_path_iter = iter(random.sample(self.text_box_image_paths, len(self.text_box_image_paths)))
            return self.random_text_box_image_path

    @property
    def random_text_box_hue(self) -> int:
        """ランダムなテキストボックスの色相を取得

        Returns:
            int: テキストボックスの色相
        """
        return random.randrange(self.text_box_hue_min, self.text_box_hue_max + 1, self.text_box_hue_step)
    
    @property
    def random_text_box_saturation(self) -> int:
        """ランダムなテキストボックスの彩度を取得

        Returns:
            int: テキストボックスの彩度
        """
        return random.randrange(self.text_box_saturation_min, self.text_box_saturation_max + 1, self.text_box_saturation_step)

    @property
    def random_text_box_color(self) -> Color:
        """ランダムなテキストボックス色を取得

        Returns:
            Color: テキストボックス色
        """
        return Color.from_hsv(HSV(self.random_text_box_hue, self.random_text_box_saturation, 100))

    @property
    def is_use_text_box_image(self) -> bool:
        """テキストボックス画像を使用するか

        呼び出すたびに結果が変わります。

        Returns:
            bool: 使用する場合は True を返します。
        """
        try:
            return next(self.use_text_box_image_iter)
        except Exception as _:
            # 一定間隔だと単調なのでシャッフルしています。
            self.use_text_box_image_iter = iter(random.sample([i == 0 for i in range(self.use_text_box_image_every_n_times)], self.use_text_box_image_every_n_times))
            return self.is_use_text_box_image

    @property
    def random_text_box_color_or_image_path(self) -> Color | str:
        """ランダムなテキストボックス色又はテキストボックス画像パスを取得

        Returns:
            Color | str: テキストボックス色又はテキストボックス画像パス
        """
        return self.random_text_box_image_path if self.is_use_text_box_image else self.random_text_box_color

    @property
    def random_text_box_top_alpha(self) -> int:
        """ランダムなテキストボックスの上部の透明度を取得

        Returns:
            int: テキストボックスの上部の透明度
        """
        return random.randrange(self.text_box_top_alpha_min, self.text_box_top_alpha_max + 1, self.text_box_top_alpha_step)

    @property
    def random_text_box_bottom_alpha(self) -> int:
        """ランダムなテキストボックスの下部の透明度を取得

        Returns:
            int: テキストボックスの下部の透明度
        """
        return random.randrange(self.text_box_bottom_alpha_min, self.text_box_bottom_alpha_max + 1, self.text_box_bottom_alpha_step)


def generate_game_image(
    background_path:str,
    canvas_size:Size2D,
    text_lines:tuple[str, ...],
    text_layout:TextLayout,
    background_width_range:Range,
    background_height_range:Range,
    text_box_color_or_image_path:Color | str,
    text_box_top_alpha:int,
    text_box_bottom_alpha:int,
    canvas_color:Color,
    is_low_quality_antialias:bool,
    is_debug_enabled:bool,
    output_directory:Path,
    idx:int,
) -> None:
    # テキスト画像
    text_layer = Image.new("RGBA", canvas_size.wh, Color.zero().rgba)
    text_drawer = ImageDraw.Draw(text_layer)

    # 低品質アンチエイリアスの作成には修飾文字を除いたテキストが必要
    if is_low_quality_antialias:
        text_alpha_layer = Image.new("L", canvas_size.wh)
        text_alpha_drawer = ImageDraw.Draw(text_alpha_layer)

    # 文字領域座標リスト
    character_bboxes:list[BoundingBox] = []

    # テキスト描画位置
    # anchor="ls" に合わせています。
    text_pos = Int2(text_layout.get_outline_weight() + text_layout.get_shadow_offset().x, canvas_size.height//2)

    # 低品質アンチエイリアス用の透明度情報
    if is_low_quality_antialias:
        text_alpha = np.zeros(canvas_size.hw, np.uint8)

    # 1行ずつ文字描画
    for text_line in text_lines:
        # 文字描画
        draw_text_layout(text_drawer, text_pos, text_line, text_layout, ColorMethod.ALPHA)

        # 文字領域の取得
        calc_character_bboxes(text_pos, text_line, text_layout.font, text_layout.anchor, text_layout.median_bbox, character_bboxes)

        # 低品質アンチエイリアスな透明度マップの作成
        if is_low_quality_antialias:
            # 文字修飾を除いたテキスト描画
            draw_simple_text(text_alpha_drawer, text_pos, text_line, text_layout.font, 255, text_layout.anchor)

            # テキストのバウンディングボックスを作成
            info = create_text_bounding_box_info(text_line, text_pos, text_layout, canvas_size)

            # 修飾を除いた透明度
            alpha = np.array(text_alpha_layer.crop(*info.bbox_decoration))

            # 修飾を含んだ透明度
            decoration_alpha = np.array(text_layer.crop(*info.bbox_decoration))[..., 3]

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

            if True:
                cv2.imshow("", np.vstack([alpha, decoration_alpha, low_quality_alpha]))
                cv2.waitKey()
                cv2.imshow("", np.zeros_like(alpha))
                cv2.waitKey(1)

        # 改行処理
        text_pos.y += text_layout.font_size + text_layout.spacing

    # 文字領域の lt, rt, lb, rb からテキスト領域を算出
    text_bbox = BoundingBox(*np.min([[bbox.xmin, bbox.ymin] for bbox in character_bboxes], axis=0).tolist(),
                            *np.max([[bbox.xmax, bbox.ymax] for bbox in character_bboxes], axis=0).tolist())

    # テキスト領域に文字修飾(縁取りや影)を適用
    apply_font_decoration(text_bbox, text_layout.get_outline_weight(), text_layout.get_shadow_weight(), text_layout.get_shadow_offset())

    # 背景画像サイズを算出
    # 背景画像サイズはテキスト領域より小さくなることはありません。
    background_size = Size2D(background_width_range.with_start(max(text_bbox.width, background_width_range.start))(),
                             background_height_range.with_start(max(text_bbox.height, background_height_range.start))())

    # 背景画像の読込とクリップ
    background_layer = Image.open(background_path).convert("RGB")
    background_layer = random_pil_crop(background_layer, background_size, Int2(background_layer.width, background_layer.height)//32)
    background_layer = np.array(background_layer)

    # テキストボックスの作成
    if isinstance(text_box_color_or_image_path, str):
        text_box_layer = Image.open(text_box_color_or_image_path).convert("RGB")
        text_box_layer = random_pil_crop(text_box_layer, background_size, Int2(text_box_layer.width, text_box_layer.height)//32)
    else:
        text_box_layer = np.full((*background_size.hw, 3), text_box_color_or_image_path.rgb, np.uint8)

    # テキストボックスの透明度を作成
    text_box_alpha = create_gradient_alpha(background_size, text_box_top_alpha, text_box_bottom_alpha)

    # 1. テキストボックスと背景を合成
    composite_layer = alpha_composite(text_box_layer, background_layer, text_box_alpha)

    # テキスト領域を切り抜き
    text_layer = text_layer.crop(tuple(text_bbox))

    # 文字色と文字透明度に分離
    if not is_low_quality_antialias:
        text_layer, text_alpha = np.dsplit(np.array(text_layer), (3, ))
    else:
        text_layer, _ = np.dsplit(np.array(text_layer), (3, ))
        text_alpha = text_alpha[text_bbox.hwslice][..., np.newaxis]

    # テキストと背景の貼り付け位置 (右下合わせ)
    text_paste_pos = Int2(background_size.width - text_bbox.width, background_size.height - text_bbox.height)

    # 貼り付け位置を中央に寄せる
    if text_paste_pos.x != 0:
        text_paste_pos.x = random.randrange(int(text_paste_pos.x * 0.3), int(text_paste_pos.x * 0.7) + 1)
    if text_paste_pos.y != 0:
        text_paste_pos.y = random.randrange(int(text_paste_pos.y * 0.3), int(text_paste_pos.y * 0.7) + 1)

    # テキストと背景の合成範囲
    xslice = slice(text_paste_pos.x, text_paste_pos.x + text_bbox.width, 1)
    yslice = slice(text_paste_pos.y, text_paste_pos.y + text_bbox.height, 1)

    # 2. テキストと背景を合成
    composite_layer[yslice, xslice] = alpha_composite(composite_layer[yslice, xslice], text_layer, 1.0 - text_alpha / 255.0)

    # キャンバス(単色)の作成
    image_layer = np.full((*canvas_size.hw, 3), canvas_color.rgb, np.uint8)

    # 背景とキャンバスの貼り付け位置
    background_paste_pos = Int2((canvas_size.width - background_size.width)//2, (canvas_size.height - background_size.height)//2)

    # 背景とキャンバスの貼り付け範囲
    xslice = slice(background_paste_pos.x, background_paste_pos.x + background_size.width, 1)
    yslice = slice(background_paste_pos.y, background_paste_pos.y + background_size.height, 1)

    # 3. 背景をキャンバスに貼り付け
    image_layer[yslice, xslice] = composite_layer

    # 貼り付けでズレた文字領域の更新
    offset = text_paste_pos + background_paste_pos
    character_bboxes = [
        BoundingBox(bbox.xmin + offset.x - text_bbox.xmin,
                    bbox.ymin + offset.y - text_bbox.ymin,
                    bbox.xmax + offset.x - text_bbox.xmin,
                    bbox.ymax + offset.y - text_bbox.ymin)
        for bbox in character_bboxes
    ]

    image_layer = cv2.cvtColor(image_layer, cv2.COLOR_RGB2BGR)

    if not is_debug_enabled:
        cv2.imwrite(str(output_directory / f"image_{idx}.jpg"), image_layer)

        with open(str(output_directory / f"bboxes_{idx}.pkl"), mode="wb") as f:
            pickle.dump(character_bboxes, f, -1)
    else:
        for bbox in character_bboxes:
            cv2.rectangle(image_layer, bbox.xymin, bbox.xymax, (255, 0, 0), 1)

        cv2.imshow("", image_layer)
        cv2.waitKey()
