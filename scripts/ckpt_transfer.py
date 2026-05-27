import torch
from src.utils import get_substatedict


def m2v_to_ad(ckpt):
    rules = {
        "attn1": "attention_blocks.0",
        "attn2": "attention_blocks.1",
        "norm1": "norms.0",
        "norm2": "norms.1",
        "norm3": "ff_norm",
        "processor.pos_encoder.pe": "pos_encoder.pe",
    }
    new_ckpt = {}
    for key, value in ckpt.items():
        new_key = key
        if "motion_modules" in key:
            prefix, endfix = key.split("motion_modules")
            endfix = endfix.split(".")
            endfix.insert(2, "temporal_transformer")
            endfix = ".".join(endfix)
            new_key = "motion_modules".join([prefix, endfix])
            for name in rules:
                if name in new_key:
                    new_key = new_key.replace(name, rules[name])
        new_ckpt[new_key] = ckpt[key]
    return new_ckpt


def gx_mix_to_m2v(ckpt):
    new_ckpt = {}
    for key, value in ckpt.items():
        if "to_out_ff" in key:
            key_names = key.split(".")
            new_key = "lamp_modules." + ".".join(key_names[:-2]).replace(".", "-") + ".to_out_mix." + key_names[-1]
            new_ckpt[new_key] = ckpt[key]
        elif "to_q_ff" in key:
            key_names = key.split(".")
            new_key = "lamp_modules." + ".".join(key_names[:-2]).replace(".", "-") + ".to_q_mix." + key_names[-1]
            new_ckpt[new_key] = ckpt[key]
        else:
            new_ckpt[key] = ckpt[key]
    return new_ckpt


ckpting_model = torch.load("exps/i2v_svd_vcg_edm_minsnr/checkpoints/checkpoint-24000/unet/pytorch_model.ckpt")
new_ckpt = {}
for key, value in ckpting_model.items():
    new_key = key
    if "_checkpoint_wrapped_module" in key:
        new_key = new_key.replace("_checkpoint_wrapped_module.", "")
    new_ckpt[new_key] = ckpting_model[key]
torch.save(new_ckpt, "exps/i2v_svd_vcg_edm_minsnr/checkpoints/checkpoint-24000/unet/pytorch_model_trans.ckpt")


# gx_ckpt = torch.load("/share_interns/guoxun/code/developing/AnimateDiff/models/i2v_module/i2v_module_mixedip_newest.pth")
# gx_ckpt = gx_mix_to_m2v(gx_ckpt)
# torch.save(gx_ckpt, "/group/zhengmingwu/diffusion-ckpts/gx_ckpt_to_m2v_i2v_module_mixedqip_newest.ckpt")

# m2v_ckpt = torch.load("/group/zhengmingwu/m2v-diffusers/exps/ad_sd15_v2_fix--t2v--vcg_all/ckpt/checkpoint-8320000/pytorch_model.bin")
# unet_m2v = get_substatedict('unet', m2v_ckpt)
# ad_unet = m2v_to_ad(unet_m2v)
# # filter out motion_module
# motion_module = {key: value for key, value in ad_unet.items() if 'motion_modules' in key}
# ad_unet_official = torch.load("/group/ckpt/diffusers/animatediff/mm_sd_v15_v2.ckpt")
# print(set(ad_unet_official.keys())==set(motion_module.keys()))
# for key, value in motion_module.items():
#     if value.shape != ad_unet_official[key].shape:
#         print(key, value.shape, ad_unet_official[key].shape)
# # torch.save(motion_module, "/group/zhengmingwu/diffusion-ckpts/mm_sd_v15_v2_m2v8320000.ckpt")
