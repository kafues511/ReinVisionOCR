import numpy as np
import numpy.typing as npt
import math
import cv2
from itertools import product

from reinlib.types.rein_size2d import Size2D
from reinlib.types.rein_bounding_box import BoundingBox


__all__ = [
    "region_to_bboxes",
    "GaussianGenerator",
]


def region_to_bboxes(
    region:npt.NDArray[np.float64],
    binary_threshold:float,
    area_threshold:int,
    gradation_threshold:float,
) -> list[BoundingBox]:
    """ヒートマップ・領域画像から文字領域座標を算出

    required_areaは2値化されたregionを元に判定するため、binary_thresholdが緩いと偽領域検出しやすい。

    偽検出を防ぐために、required_gradientで2値化されていないregionの階調を元に判定している。

    文字領域は加工しやすいようにtupleで囲んでいません。

    Args:
        region (npt.NDArray[np.float64]): ヒートマップ・領域画像
        binary_threshold (float): ヒートマップに適用する2値化閾値
        area_threshold (int): 文字領域として必要な面積(単位:ピクセル数)
        gradation_threshold (float): 文字領域として必要な階調

    Returns:
        list[BoundingBox]: 文字領域(4頂点)リスト
    """
    _, text_score = cv2.threshold(region, binary_threshold, 1.0, cv2.THRESH_BINARY)
    text_score = np.clip(text_score * 255, 0, 255).astype(np.uint8)

    # クラスタリング
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(text_score, connectivity=4)

    bboxes:list[BoundingBox] = []

    for k in range(1, retval):
        x, y, w, h, area = stats[k]

        # 文字領域として必要な面積があるか
        if area < area_threshold:
            continue

        # 偽検出防止用に2値化前のヒートマップ画像に必要な階調が含まれているか判定
        if np.max(region[labels==k]) < gradation_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(region.shape, dtype=np.uint8)
        segmap[labels==k] = 255

        # https://github.com/clovaai/CRAFT-pytorch/issues/63
        niter = int(math.sqrt(area * min(w, h) / (w * h)) * 2)

        # boundary check
        xmin, ymin = max(0, x - niter), max(0, y - niter)
        xmax, ymax = min(x + w + niter + 1, region.shape[1]), min(y + h + niter + 1, region.shape[0])

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[ymin:ymax, xmin:xmax] = cv2.dilate(segmap[ymin:ymax, xmin:xmax], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = np.intp(cv2.boxPoints(rectangle) * 2)

        bboxes.append(BoundingBox(*np.min(box, axis=0).tolist(), *np.max(box, axis=0).tolist()))

    return bboxes


class GaussianGenerator:
    """ヒートマップ・領域画像の生成器
    """
    def __init__(
        self,
        kernel_size:Size2D = Size2D(64, 64),
        distance_ratio:float = 5.0,
    ) -> None:
        """コンストラクタ

        Args:
            kernel_size (Size2D, optional): ヒートマップの画像サイズ. Defaults to Size2D(64, 64).
            distance_ratio (float, optional): ヒートマップの半径を決める係数. Defaults to 5.0.
        """
        self.kernel_size = kernel_size
        self.distance_ratio = distance_ratio
        self.gaussian2d = self.isotropic_gaussian_heatmap(kernel_size=kernel_size, distance_ratio=distance_ratio)

    @staticmethod
    def isotropic_gaussian_heatmap(
        kernel_size:Size2D = Size2D(64, 64),
        distance_ratio:float = 5.0,
    ) -> npt.NDArray[np.float64]:
        """ヒートマップの作成

        Args:
            kernel_size (Size2D, optional): ヒートマップの画像サイズ. Defaults to Size2D(64, 64).
            distance_ratio (float, optional): ヒートマップの半径を決める係数. Defaults to 5.0.

        Returns:
            npt.NDArray[np.float64]: ヒートマップ
        """
        w, h = kernel_size.wh
        half_w, half_h = w * 0.5, h * 0.5
        # 楕円ではなく円を作成するため、どちらか長い辺を選択
        half_max = max(half_w, half_h)
        gaussian2d_heatmap = np.zeros((h, w), np.float64)

        for y, x in product(range(h), range(w)):
            # ピクセル座標から中心座標のユークリッド距離を算出
            distance_from_center = np.linalg.norm(np.array([y - half_h, x - half_w]))
            # 算出された距離にdraioを適用
            distance_from_center = distance_ratio * distance_from_center / half_max
            scaled_gaussian_prob = math.exp(-0.5 * (distance_from_center ** 2))
            gaussian2d_heatmap[y, x] = np.clip(scaled_gaussian_prob * 255, 0, 255)

        return gaussian2d_heatmap

    @staticmethod
    def perspective_transform(
        image:npt.NDArray[np.float64],
        bbox:BoundingBox,
    ) -> npt.NDArray[np.float64]:
        """射影変換

        各頂点は変換後の画像サイズを求めるための情報です。

        射影変換はArea補間で行われます。
        変換後の画像サイズが入力画像より小さい場合の品質劣化を抑制しています。

        Args:
            image (npt.NDArray[np.float64]): 射影変換が行われるグレースケール画像
            bbox (BoundingBox): 変換後のバウンディングボックス

        Returns:
            npt.NDArray[np.float64]: 変換後の画像
        """
        assert len(image.shape) == 2, "Perspective transform target requires grayscale."

        src_h, src_w = image.shape
        dst_h, dst_w = bbox.hw

        src_pts = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)
        dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        image = cv2.warpPerspective(image, matrix, (dst_w, dst_h), flags=cv2.INTER_AREA)

        return image

    def __call__(
        self,
        image_size:Size2D,
        bboxes:list[BoundingBox],
    ) -> npt.NDArray[np.float64]:
        """バウンディングボックスリストを元にヒートマップ・領域画像を作成

        バウンディングボックスリストの座標は、0以上で且つ画像サイズ未満である必要があります。

        Args:
            image_size (Size2D): 画像サイズ
            bboxes (list[BoundingBox]): バウンディングボックスリスト

        Returns:
            npt.NDArray[np.float64]: ヒートマップ・領域画像
        """
        image = np.zeros(image_size.hw, dtype=np.float64)
        g2dheatmap = self.gaussian2d.copy()

        for bbox in bboxes:
            image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax] += self.perspective_transform(g2dheatmap, bbox)

        return image
