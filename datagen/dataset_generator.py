from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager

from reinlib.types.rein_stage_type import StageType
from reinlib.utility.rein_dataset_generator import DatasetGeneratorAbstract

from datagen.image_generator import *


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
        with Manager() as manager:
            self.config.setup(manager)

            with ProcessPoolExecutor(self.config.max_workers) as executor:
                for stage_type in (StageType.TRAIN, ):
                    stage_directory = self.config.create_stage_directory(stage_type)

                    futures:list[Future] = []

                    futures += [
                        executor.submit(
                            generate_word_image,
                            *parameter,
                            stage_directory,
                            idx,
                        )
                        for idx, parameter in enumerate(self.config.create_word_parameters())
                    ]

                    futures += [
                        executor.submit(
                            generate_sentence_image,
                            *parameter,
                            stage_directory,
                            idx,
                        )
                        for idx, parameter in enumerate(self.config.create_sentence_parameters(), len(futures))
                    ]

                    futures += [
                        executor.submit(
                            generate_game_image,
                            *parameter,
                            stage_directory,
                            idx,
                        )
                        for idx, parameter in enumerate(self.config.create_game_parameters(), len(futures))
                    ]

                    if self.config.is_tqdm_enabled:
                        from tqdm import tqdm
                        with tqdm(futures, desc=f"{stage_type}", postfix=f"v_num={self.config.output_version}") as pbar:
                            for future in pbar:
                                future.result()
                    else:
                        for future in futures:
                            future.result()

                    futures.clear()

            self.config.close()
