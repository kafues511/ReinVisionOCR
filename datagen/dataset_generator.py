from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from reinlib.types.rein_stage_type import StageType
from reinlib.utility.rein_dataset_generator import DatasetGeneratorAbstract

from datagen.image_generator import GenerateConfig
from datagen.character_image_generator import generate_character_image


__all__ = [
    "DatasetGenerator",
]


class DatasetGenerator(DatasetGeneratorAbstract):
    """データセット生成クラス
    """
    @property
    def generate_config_cls(self) -> GenerateConfig:
        """生成設定クラス

        Returns:
            GenerateConfigBase: 生成設定クラス
        """
        return GenerateConfig

    @property
    def config(self) -> GenerateConfig:
        """生成設定を取得

        Returns:
            GenerateConfig: _description_
        """
        return self.__config

    @config.setter
    def config(self, new_config:GenerateConfig) -> None:
        """生成設定をセット

        Args:
            new_config (GenerateConfig): 生成設定
        """
        self.__config = new_config

    def generate_impl(self) -> None:
        """データセット生成
        """
        with ThreadPoolExecutor(self.config.max_workers) as executor:
            for stage_type in (StageType.TRAIN, ):
                stage_directory = self.config.create_stage_directory(stage_type)

                # NOTE:
                # データ数が膨大な場合はひとつのスレッドで複数の画像を作成した方が高速です。
                # これはスレッド作成に時間が掛かるためです。
                futures = [
                    executor.submit(
                        generate_character_image,
                        *parameter,
                        stage_directory,
                        idx,
                    )
                    for idx, parameter in enumerate(self.config.create_dataset_parameters(stage_type))
                ]

                if self.config.is_tqdm_enabled:
                    with tqdm(futures, desc=f"{stage_type}", postfix=f"v_num={self.config.output_version}") as pbar:
                        for future in pbar:
                            future.result()
                else:
                    for future in futures:
                        future.result()

                # clear memory
                futures.clear()
