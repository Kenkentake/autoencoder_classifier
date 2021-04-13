from pytorch_lightning import LightningModule

from models.conv_autoencoder_model import ConvAutoEncoderCLFModel
from models.simple_cnn_model import SimpleCNNModel

def get_model(
    args,
    device: str,
    hparams: dict,
    trial=None,
) -> LightningModule:
    model_type = args.TRAIN.MODEL_TYPE.lower()

    if model_type == 'simple_cnn':
        model = SimpleCNNModel(
            args=args,
            device=device,
            hparams=hparams,
            in_channel=args.DATA.INPUT_DIM,
            out_channel=len(args.DATA.CLASSES),
            trial=trial,
        )
    elif model_type == 'conv_autoencoder':
        model = ConvAutoEncoderCLFModel(
            args=args,
            device=device,
            hparams=hparams,
            in_channel=args.DATA.INPUT_DIM,
            out_channel=len(args.DATA.CLASSES),
            trial=trial,
        )
    else:
        raise NotImplementedError()

    return model
