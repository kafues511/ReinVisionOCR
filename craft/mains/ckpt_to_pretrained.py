import os
from copy import copy
from pathlib import Path
import torch
import zipfile

from reinlib.utility.rein_files import try_readlink, get_sort_hash
from reinlib.utility.rein_torch import find_minimum_loss_checkpoint


def my_app(
    checkpoint_path_or_directory:str,
    output_directory:str,
    model_name:str,
) -> None:
    """チェックポイントから学習済みモデルを作成

    Args:
        checkpoint_path_or_directory (str): チェックポイントパスもしくはチェックポイントが保存されたディレクトリ
        output_directory (str): 出力先のディレクトリ
        model_name (str): モデル名
    """
    # ディレクトリが指定された場合は損失が最小のモデルから学習済みモデルを作成します。
    if Path(checkpoint_path_or_directory).suffix != ".ckpt":
        checkpoint_path = find_minimum_loss_checkpoint(checkpoint_path_or_directory)
    else:
        checkpoint_path = checkpoint_path_or_directory

    # チェックポイントがシンボリックリンクの場合は参照元のパスを取得
    if (symlink_path:=try_readlink(checkpoint_path)) is not None:
        checkpoint_path = symlink_path

    # チェックポイント読込
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # 学習に使用したデータセットの生成設定
    dataset_config_paths = [
        Path(dataset_dir).absolute() / "config.yml"
        for dataset_dir in copy(checkpoint["datamodule_hyper_parameters"]["dataset_directories"])
    ]

    # 不必要なパラメータの削除
    for key in copy(list(checkpoint)):
        if key == "state_dict":
            for val in list(checkpoint[key]):
                checkpoint[key][val.replace("model.", "")] = checkpoint[key].pop(val)
        else:
            del checkpoint[key]

    # ナンバリング
    output_directory:Path = Path(output_directory)
    output_version = [int(dir.stem.split("_v")[1]) for dir in output_directory.glob("*.pth") if dir.name.startswith(model_name)]
    output_version = max(output_version) + 1 if len(output_version) > 0 else 0

    # 出力先の作成
    output_path = output_directory / f"{model_name}_v{output_version}.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # str to Path
    checkpoint_path:Path = Path(checkpoint_path)

    # モデルの保存
    torch.save(checkpoint, str(output_path))

    # パラメータの出力先
    output_config_path = output_directory / f"{model_name}_v{output_version}.zip"

    # 使用頻度は高くないのでzipで保存
    with zipfile.ZipFile(str(output_config_path), "a") as zf:
        # ハイパラ
        zf.write(str(checkpoint_path.parent.parent / "hparams.yaml"))

        # 学習に使用したデータセットの生成設定
        for dataset_config_path in dataset_config_paths:
            zf.write(str(dataset_config_path))

        # データセット生成の再現性を確保するために現在のリビジョンを出力
        if (sort_hash:=get_sort_hash()) is not None:
            with zf.open(f"{sort_hash}.txt", "w") as f:
                f.write(sort_hash.encode())

    # ファイルサイズ
    before_size = os.path.getsize(checkpoint_path)
    after_size = os.path.getsize(str(output_path))

    # 出力先とファイルサイズを通知
    print(str(output_path))
    print(f"File size (before > after): {before_size/1024/1024:.3f} > {after_size/1024/1024:.3f} MB")
