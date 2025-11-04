from models.predictor import DenseGCNPredictor
import torchinfo
import torch

torchinfo.summary(
    DenseGCNPredictor([64, 128, 256, 512], 98, 8),
    input_data=[
        {
            "out2": torch.randn(1, 128, 32, 32),
            "out3": torch.randn(1, 256, 16, 16),
            "out4": torch.randn(1, 512, 8, 8),
        },
    ],
)
