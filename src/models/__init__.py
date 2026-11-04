import torch
from torch import nn

from models.encdec import EncDec
from models.unet import UNet

MODELS = {"encdec": EncDec, "unet": UNet}


def build_model(name, base_channels=32, depth=4):
    return MODELS[name](base_channels=base_channels, depth=depth)


@torch.no_grad()
def shape_table(model, image_size):
    """(lag, type, output-shape C x H x W, antal parametre) i den rækkefølge lagene kaldes.

    Bruges til U-Net-tabellen i rapporten og til at tjekke, at implementeringen matcher skitsen.
    """
    rows = [("input", "", (3, image_size, image_size), 0)]
    handles = []
    for name, child in model.named_children():
        modules = child.named_children() if isinstance(child, nn.ModuleList) else [("", child)]
        for sub_name, module in modules:
            label = f"{name}.{sub_name}" if sub_name else name
            n_params = sum(p.numel() for p in module.parameters())

            def hook(module, inputs, output, label=label, n_params=n_params):
                rows.append((label, type(module).__name__, tuple(output.shape[1:]), n_params))

            handles.append(module.register_forward_hook(hook))
    model.eval()
    output = model(torch.zeros(1, 3, image_size, image_size))
    for handle in handles:
        handle.remove()
    rows.append(("output", "", tuple(output.shape[1:]), 0))
    return rows
