import os
import tyro
import warnings

from src.configs.test_configs import TestConfig, test_configs
from src.configs.train_configs import train_configs
from src.engine.trainer import TrainerConfig
# from src.utils import CONSOLE, convert_markup_to_ansi, get_current_git_commit, set_environments
from src.utils import CONSOLE, convert_markup_to_ansi, get_current_git_commit




def main(config: TrainerConfig):
    trainer = config.setup()
    trainer.test_2(mode="Test")


def entrypoint():
    method_configs = {**train_configs, **test_configs}
    AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
        tyro.conf.FlagConversionOff[tyro.extras.subcommand_type_from_defaults(defaults=method_configs)]
    ]

    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    # set_environments()
    # Avoid dataset pending
    os.environ["http_proxy"] = "http://oversea-squid1.jp.txyun:11080"
    os.environ["https_proxy"] = "http://oversea-squid1.jp.txyun:11080"
    os.environ["no_proxy"] = "localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"
    os.environ["TORCH_HOME"] = "/group/ckpt/torchhub"
    os.environ["HF_DATASETS_CACHE"] = "/video/cache/huggingface"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    entrypoint()

    '''
    python infer.py align_1B_infer --experiment-name align_1B_infer_valprompts
    '''