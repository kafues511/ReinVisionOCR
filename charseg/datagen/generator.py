from concurrent.futures import ProcessPoolExecutor

from reinlib.types.rein_stage_type import StageType
from reinlib.utility.rein_dataset_generator import DatasetGeneratorAbstract

from datagen.generate_impl import GenerateConfig
from datagen.generate_character import generate_character_image


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
            GenerateConfig: 生成設定
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
        with ProcessPoolExecutor(self.config.max_workers) as executor:
            for stage_type in (StageType.TRAIN, ):
                stage_directory = self.config.create_stage_directory(stage_type)

                futures = [
                    executor.submit(
                        generate_character_image,
                        *parameter,
                        stage_directory,
                        idx * self.config.characters_per_executor,
                    )
                    for idx, parameter in enumerate(self.config.create_dataset_parameters(stage_type))
                ]

                if self.config.is_tqdm_enabled:
                    from tqdm import tqdm
                    with tqdm(desc=f"{stage_type}", postfix=f"v_num={self.config.output_version}", total=self.config.total) as pbar:
                        for future in futures:
                            pbar.update(future.result())
                else:
                    for future in futures:
                        future.result()

                # clear memory
                futures.clear()
