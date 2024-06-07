import math
from typing import Optional
from PIL import ImageFont
from fontTools.ttLib import TTFont
from multiprocessing.managers import SyncManager, ListProxy

from reinlib.types.rein_size2d import Size2D
from reinlib.utility.rein_files import get_paths_in_directories
from reinlib.utility.rein_text_draw import calc_median_bbox
from reinlib.utility.rein_generate_config import GenerateConfigBase

from datagen.image_generator_base import *
from datagen.word_image_generator import *
from datagen.sentence_image_generator import *
from datagen.game_image_generator import *


class GenerateConfig(GenerateConfigBase):
    # 背景画像の対応拡張子
    IMAGE_SUFFIXES = tuple([".png", ".jpg", ".jpeg"])

    """生成設定クラス
    """
    def __init__(
        self,

        background_image_directories:tuple[str, ...],

        canvas_width:int,
        canvas_height:int,

        font_path:str,
        font_size:int,

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

        is_low_quality_antialias:bool,

        brackets_list:tuple[tuple[str, str], ...],
        custom_text_list:tuple[str, ...],

        character_parameters:tuple[tuple[str, int], ...],

        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # 背景画像パスの読込
        self.background_image_paths = get_paths_in_directories(background_image_directories, self.IMAGE_SUFFIXES)

        # 出力画像サイズ
        self.canvas_size = Size2D(canvas_width, canvas_height)

        # フォント読込と対応文字をUnicodeで取得
        self.font = ImageFont.truetype(font_path, font_size)
        self.cmap:tuple[int, ...] = tuple(TTFont(font_path).getBestCmap())

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

        # 低品質アンチエイリアスを使用するか
        self.is_low_quality_antialias = is_low_quality_antialias

        # 縁取りと影の両方が有効な場合には 透明度を同期 させる必要があります。
        if is_low_quality_antialias and (is_outline and is_shadow):
            assert is_apply_same_alpha, "is_apply_same_alpha must be True to enable low quality anti-aliasing when outlines and shadows are enabled."

        # 括弧類
        self.brackets_list = tuple(tuple(brackets) for brackets in brackets_list)

        # カスタムテキスト
        self.custom_text_list = tuple(custom_text_list)

        # 生成文字リスト
        characters:list[str] = []

        # 生成文字の読込
        for character_parameter in character_parameters:
            self.register_characters(*character_parameter, characters)

        # 文字列化
        self.characters = "".join(characters)

        # 生成文字リストから文字領域の中央値のバウンディングボックスを算出
        # NOTE: 文字描画位置の算出に使用します。
        self.median_bbox = calc_median_bbox(set(self.characters), self.font)

        # 文字修飾を考慮した最終的な文字サイズ
        character_size = self.font.size + (outline_weight if is_outline else 0) + (shadow_weight + abs(self.shadow_offset.x) if is_shadow else 0)

        # 1行あたりの最大文字数を取得
        # 横幅をピッタシにすると random.randrange(start, stop, step) を使う前に start != stop 判定が必要になる
        # 判定入れるのが面倒なので -1 pixel しておく
        self.characters_per_line = math.floor((self.canvas_size.width - 1) / character_size)

        # 単語画像設定
        self.word_image_config = WordImageConfig(**kwargs["word_image_config"])
        self.word_image_config.clamp_word_length(self.canvas_size.width, self.font.size)

        # 文章画像設定
        self.sentence_image_config = SentenceImageConfig(**kwargs["sentence_image_config"])
        self.sentence_image_config.clamp_sentence_length(self.canvas_size.width, self.font.size)

        # ゲーム画像設定
        self.game_image_config = GameImageConfig(**kwargs["game_image_config"])

    def register_characters(
        self,
        characters_or_path:str,
        repeat:int,
        out_characters:list[str],
    ) -> None:
        """文字列の読込

        Args:
            characters_or_path (str): 文字列もしくはテキストファイルパス
            repeat (int): 文字列の繰り返し数
            out_characters (list[str]): 読み込んだ文字列の格納先
        """
        # read text file.
        if characters_or_path.rfind(".txt") != -1:
            with open(characters_or_path, mode="r", encoding="utf-8") as f:
                characters_or_path = f.read()

        # half to full & remove empty
        characters_or_path = characters_or_path.translate(HALF2FULL_TABLE)
        characters_or_path = characters_or_path.translate(EMPTY_TABLE)

        # 未対応文字を除外
        characters_or_path = tuple([char for char in characters_or_path if ord(char) in self.cmap])

        # apply repeat.
        out_characters.extend(characters_or_path * repeat)

    def setup(self, manager:SyncManager) -> None:
        """マルチプロセス間処理のセットアップ

        Args:
            manager (SyncManager): SyncManager
        """
        # プロセス間で共有の文字リスト
        self.character_list_proxy:ListProxy = manager.list()

        # character_list用のlock
        self.lock = manager.Lock()
    
    def close(self) -> None:
        """終了処理
        """
        self.character_list_proxy[:] = []  # clear

    @property
    def background_image_path(self) -> str:
        """背景画像パスを取得

        Returns:
            str: 背景画像パス
        """
        try:
            return str(next(self.background_imaeg_path_iter))
        except Exception as _:
            self.background_imaeg_path_iter = iter(random.sample(self.background_image_paths, len(self.background_image_paths)))
            return self.background_image_path

    @property
    def random_text_color(self) -> tuple[int, int, int]:
        """ランダムな文字色を取得

        Returns:
            tuple[int, int, int]: ランダムな文字色 (RGB)
        """
        return Color.from_hsv(HSV(self.hue_range(), self.saturation_range(), 100)).rgb

    @property
    def brackets(self) -> tuple[str, str]:
        """括弧を取得

        Returns:
            tuple[str, str]: 括弧
        """
        try:
            return next(self.brackets_iter)
        except Exception as _:
            self.brackets_iter = iter(self.brackets_list)
            return self.brackets

    @property
    def custom_text(self) -> str:
        """カスタムテキストを取得

        Returns:
            str: カスタムテキスト
        """
        try:
            return next(self.custom_text_iter)
        except Exception as _:
            self.custom_text_iter = iter(self.custom_text_list)
            return self.custom_text

    @property
    def sentence_lengths(self) -> tuple[int, ...]:
        """文章の長さリストを取得

        Returns:
            tuple[int, ...]: 文章の長さリスト
        """
        return [
            self.sentence_image_config.random_sentence_length
            for _ in range(self.sentence_image_config.max_plots)
        ]

    @property
    def sentence_custom_text_list(self) -> tuple[Optional[str], ...]:
        """文章のテキストリストを取得

        Returns:
            tuple[Optional[str], ...]: 文章のテキストリスト
        """
        return [
            self.custom_text if self.sentence_image_config.is_insert_custom_text else None
            for _ in range(self.sentence_image_config.max_plots)
        ]

    @property
    def sentence_brackets_list(self) -> tuple[Optional[tuple[str, str]], ...]:
        """文章の括弧類リストを取得

        Returns:
            tuple[Optional[tuple[str, str]], ...]: 文章の括弧類リスト
        """
        return [
            self.brackets if self.sentence_image_config.is_insert_brackets else None
            for _ in range(self.sentence_image_config.max_plots)
        ]

    @property
    def sentence_text_layouts(self) -> tuple[TextLayout, ...]:
        """文章の文字描画設定リストを取得

        Returns:
            tuple[TextLayout, ...]: 文字描画設定リスト
        """
        return [
            self.create_text_layout(self.sentence_image_config.is_use_random_color)
            for _ in range(self.sentence_image_config.max_plots)
        ]

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
    def text(self) -> str:
        """テキストを取得

        Returns:
            str: テキスト
        """
        return "".join(self.character for _ in range(self.game_image_config.random_text_length))

    @property
    def text_lines(self) -> tuple[str, ...]:
        """行ごとに分割したテキストを取得

        Returns:
            tuple[str, ...]: 行ごとに分割されたテキスト
        """
        # ランダムな長さのテキストを取得
        text = self.text

        # カスタムテキストの挿入
        if self.game_image_config.is_insert_custom_text:
            text = insert_text(text, self.custom_text, random.randrange(0, len(text) + 1))

        # 改行後の先頭に付ける文字
        text_head = ""

        # 囲い文字の挿入
        if self.game_image_config.is_insert_brackets:
            begin_bracket, end_bracket = self.brackets
            text = f"{begin_bracket}{text}{end_bracket}"

            # 先頭の囲い文字が1文字以上なら改行後の1文字目を空白にする
            if len(begin_bracket) > 0:
                text_head = "　"

        # 1行あたりの最大文字数を取得
        chars_per_line = self.characters_per_line

        # chars_per_lineと同値の場合、ゲームではあり得ない改行がされるので文字数を削る
        if len(text) == chars_per_line:
            text = text[:-1]

        # 改行後の先頭に挿入する文字列の文字数
        text_head_length = len(text_head)

        # 何行生成されるか
        num_line = math.ceil(len(text) / chars_per_line)
        num_line = math.ceil((num_line * text_head_length + len(text)) / chars_per_line)

        # テキストを行ごとに分割
        text_lines:list[str] = []
        for row in range(num_line):
            if row == 0:
                begin = row * chars_per_line
                end = min(len(text), row * chars_per_line + chars_per_line)
                text_lines.append(text[begin:end])
            else:
                _row = row - 1
                _chars_per_line = chars_per_line - text_head_length
                # 改行後の先頭に挿入する文字列を考慮してスライス
                begin = chars_per_line + _row * _chars_per_line
                end = min(len(text), chars_per_line + _row * _chars_per_line + _chars_per_line)
                text_lines.append(f"{text_head}{text[begin:end]}")

        return tuple(text_lines)

    def create_text_layout(self, is_random_text_color:bool = False, spacing:int = 0) -> TextLayout:
        """テキストレイアウトの作成

        Args:
            is_random_text_color (bool, optional): 文字色にランダム色を使用するか. Defaults to False.
            spacing (int, optional): 行間サイズ. Defaults to 0.

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
            spacing=spacing,
            median_bbox=self.median_bbox,
        )

    def create_word_drawer_parameters(self) -> tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...]:
        """文字描画パラメータリストの作成

        Returns:
            tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...]: 文字描画パラメータリスト
        """
        parameters:list[tuple[int, Optional[tuple[str, str]], Optional[str], TextLayout]] = []

        for _ in range(self.word_image_config.max_plots):
            word_length = self.word_image_config.random_word_length
            custom_text = self.custom_text if self.word_image_config.is_insert_custom_text else None
            brackets = self.brackets if self.word_image_config.is_insert_brackets else None
            text_layout = self.create_text_layout(self.word_image_config.is_use_random_color)
            parameters.append((word_length, custom_text, brackets, text_layout))

        return tuple(parameters)

    def create_sentence_drawer_parameters(self) -> tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...]:
        """文字描画パラメータリストの作成

        Returns:
            tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...]: 文字描画パラメータリスト
        """
        parameters:list[tuple[int, Optional[tuple[str, str]], Optional[str], TextLayout]] = []

        for _ in range(self.sentence_image_config.max_plots):
            word_length = self.sentence_image_config.random_sentence_length
            custom_text = self.custom_text if self.sentence_image_config.is_insert_custom_text else None
            brackets = self.brackets if self.sentence_image_config.is_insert_brackets else None
            text_layout = self.create_text_layout(self.sentence_image_config.is_use_random_color)
            parameters.append((word_length, custom_text, brackets, text_layout))

        return tuple(parameters)

    def create_word_parameters(self) -> list[tuple[Lock, str, Size2D, Size2D, tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...], str, ListProxy, bool, bool]]:
        """単語をプロットした画像の生成パラメータリストを作成

        Returns:
            list[tuple[Lock, str, Size2D, Size2D, tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...], str, ListProxy, bool, bool]]: パラメータリスト
        """
        return [
            (
                self.lock,
                self.background_image_path,
                self.canvas_size,
                self.word_image_config.word_margin,
                self.create_word_drawer_parameters(),
                self.characters,
                self.character_list_proxy,
                self.is_low_quality_antialias,
                self.is_debug_enabled,
            )
            for _ in range(self.word_image_config.get_total())
        ]

    def create_sentence_parameters(self) -> list[tuple[Lock, str, Size2D, Size2D, tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...], str, ListProxy, bool, bool]]:
        """文章をプロットした画像の生成パラメータリストを作成

        Returns:
            list[tuple[Lock, str, Size2D, Size2D, tuple[tuple[int, Optional[str], Optional[tuple[str, str]], TextLayout], ...], str, ListProxy, bool, bool]]: パラメータリスト
        """
        return [
            (
                self.lock,
                self.background_image_path,
                self.canvas_size,
                self.sentence_image_config.sentence_margin,
                self.create_sentence_drawer_parameters(),
                self.characters,
                self.character_list_proxy,
                self.is_low_quality_antialias,
                self.is_debug_enabled,
            )
            for _ in range(self.sentence_image_config.get_total())
        ]

    def create_game_parameters(self) -> list[tuple[str, Size2D, tuple[str, ...], TextLayout, Range, Range, Color | str, int, int, Color, bool, bool]]:
        """テキストをゲーム風にプロットした画像の生成パラメータリストを作成

        Returns:
            list[tuple[str, Size2D, tuple[str, ...], TextLayout, Range, Range, Color | str, int, int, Color, bool, bool]]:: パラメータリスト
        """
        return [
            (
                self.background_image_path,
                self.canvas_size,
                self.text_lines,
                self.create_text_layout(self.game_image_config.is_use_random_color, self.game_image_config.spacing),
                self.game_image_config.background_width_range,
                self.game_image_config.background_height_range,
                self.game_image_config.random_text_box_color_or_image_path,
                self.game_image_config.random_text_box_top_alpha,
                self.game_image_config.random_text_box_bottom_alpha,
                self.game_image_config.canvas_color,
                self.is_low_quality_antialias,
                self.is_debug_enabled,
            )
            for _ in range(self.game_image_config.get_total())
        ]
