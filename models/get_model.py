from pytorch_lightning import LightningModule

from models.conv_ae_with_cnn_model import ConvAEWithCNNCLFModel
from models.conv_autoencoder_model import ConvAutoEncoderCLFModel
from models.simple_cnn_model import SimpleCNNModel

def get_model(
    args,
    run_id: str,
    tmp_results_dir: str,
    device: str,
    hparams: dict,
    trial=None
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
            run_id=run_id,
            tmp_results_dir=tmp_results_dir
        )
    elif model_type == 'conv_ae_with_cnn':
        model = ConvAEWithCNNCLFModel(
            args=args,
            device=device,
            hparams=hparams,
            in_channel=args.DATA.INPUT_DIM,
            out_channel=len(args.DATA.CLASSES),
            trial=trial,
            run_id=run_id,
            tmp_results_dir=tmp_results_dir
        )
    else:
        raise NotImplementedError()

    return model
