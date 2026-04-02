import torch
import os
from models.resnet import ResNet1D, ResNet1DMoE

# ==================== 配置 ====================
WEIGHTS_DIR = "weights"
OUT_DIR = "onnx_models"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_S = {
    'base_filters': 32, 'kernel_size': 3, 'stride': 2,
    'groups': 1, 'n_block': 18, 'n_classes': 512, 'n_experts': 3
}
CONFIG_P = {
    'base_filters': 32, 'kernel_size': 3, 'stride': 2,
    'groups': 1, 'n_block': 18, 'n_classes': 512,
}
CONFIG_SVRI = {
    'base_filters': 32, 'kernel_size': 3, 'stride': 2,
    'groups': 1, 'n_block': 18, 'n_classes': 512,
}

# ==================== 加载模型 ====================
def load_model(model_type):
    if model_type == "s":
        model = ResNet1DMoE(in_channels=1, **CONFIG_S)
        fname = "papagei_s.pt"
    elif model_type == "p":
        model = ResNet1D(in_channels=1, **CONFIG_P)
        fname = "papagei_p.pt"
    elif model_type == "svri":
        model = ResNet1D(in_channels=1, **CONFIG_SVRI, use_mt_regression=False, use_projection=False)
        fname = "papagei_s_svri.pt"

    ckpt = torch.load(os.path.join(WEIGHTS_DIR, fname), map_location="cpu")
    new_sd = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("_orig_mod."):
            k = k[9:]
        new_sd[k] = v
    model.load_state_dict(new_sd)
    model.eval()
    return model, fname.replace(".pt", "")

# ==================== 导出 ONNX ====================
def export(model, name):
    dummy = torch.randn(1, 1, 1250)
    onnx_path = os.path.join(OUT_DIR, f"{name}.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"✅ 导出成功：{onnx_path}")

# ==================== 执行 ====================
for typ in ["s", "p", "svri"]:
    model, name = load_model(typ)
    export(model, name)

print("\n🎉 全部 ONNX 导出完成！")
