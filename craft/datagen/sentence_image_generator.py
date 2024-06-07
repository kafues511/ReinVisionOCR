from dataclasses import dataclass
import math

from datagen.image_generator_base import *
from datagen.word_image_generator import generate_word_image


@dataclass
class SentenceImageConfig(CommonImageConfig):
    """文章画像設定
    """
    # 文章の最小、最大の長さ
    sentence_length_min:int
    sentence_length_max:int
    sentence_length_step:int

    # 文章の最大プロット数
    max_plots:int

    # 文章の余白サイズ
    # キャンバスの余白ではなく、文章をプロットした領域の余白です。
    sentence_margin_width:int
    sentence_margin_height:int

    def clamp_sentence_length(self, canvas_width:int, font_size:int) -> None:
        """文章の長さをクランプする

        Args:
            canvas_width (int): キャンバスの横幅
            font_size (int): 1文字あたりの最大横幅（おそらくはフォントサイズ）
        """
        # 余白を考慮した最大可能な最大文字数（横幅）
        sentence_length_max = math.floor((canvas_width - self.sentence_margin_width * 2) / font_size)
        if self.sentence_length_max > sentence_length_max:
            self.sentence_length_max = sentence_length_max

    @property
    def random_sentence_length(self) -> int:
        """ランダムな文章の長さを取得

        Returns:
            int: 文章の長さ
        """
        return random.randrange(self.sentence_length_min, self.sentence_length_max + 1, self.sentence_length_step)

    @property
    def sentence_margin(self) -> Size2D:
        """文章の余白サイズを取得

        Returns:
            Size2D: 文章の余白サイズ
        """
        return Size2D(self.sentence_margin_width, self.sentence_margin_height)


def generate_sentence_image(*args, **kwargs) -> None:
    try:
        generate_word_image(*args, **kwargs)
    except Exception as e:
        print(e)
