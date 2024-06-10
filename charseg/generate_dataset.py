from mains.generate_dataset import my_app


if __name__ == "__main__":
    """データセットの生成 (vscode)
    """
    config_path_list = [
        #r"configs\msgothic_shadow_30.yml",
        #r"configs\mtlmr3m_outline_shadow_30.yml",
    ]

    for config_path in config_path_list:
        my_app(config_path)
