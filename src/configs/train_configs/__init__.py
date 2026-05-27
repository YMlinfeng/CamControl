from .animatediff_configs import animatediff_configs
from .dit_configs import dit_configs
from .lumiere_configs import lumiere_configs
from .svd_configs import svd_configs
from .t2i_configs import t2i_configs
from .tokenizer_configs import vtokenizer_configs
from .videofusion_configs import videofusion_configs

try:
    from .personal_configs import personal_configs
except Exception as e:
    print(f"An error occurred: {e}")
    personal_configs = {}


train_configs = {
    **animatediff_configs,
    **videofusion_configs,
    **t2i_configs,
    **svd_configs,
    **lumiere_configs,
    **dit_configs,
    **vtokenizer_configs,
    **personal_configs,
}
