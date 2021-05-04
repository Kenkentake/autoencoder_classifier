from os.path import join

from yacs.config import CfgNode

from config.const import PROJECT_ROOT


_C = CfgNode()
_C.SEED = 42


# train
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.CE_CLASS_WEIGHT = [1., 1., 2.5, 1., 2.5, 1., 1., 1., 1., 2.5]
_C.TRAIN.CE_CLASS_WEIGHT = []
_C.TRAIN.DISTRIBUTED_BACKEND = 'ddp'
_C.TRAIN.GPUS = 1
_C.TRAIN.LOSS_WEIGHT = []
_C.TRAIN.LR = 0.01
_C.TRAIN.MAX_EPOCHS = 200
_C.TRAIN.MODEL_TYPE = ''
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER_TYPE = 'adam'
_C.TRAIN.SCHEDULER_TYPE = 'step_lr'
_C.TRAIN.STEP_SIZE = 5
_C.TRAIN.LOAD_WEIGHT_PATH = ''

_C.MLFLOW = CfgNode()
_C.MLFLOW.EXPERIMENT_NAME = 'Default'

_C.OPTUNA = CfgNode()
_C.OPTUNA.N_TRIALS = 2
_C.OPTUNA.TIMEOUT = 60 * 60


_C.DATA = CfgNode()
_C.DATA.CACHE_DIR = join(PROJECT_ROOT, '.data')
_C.DATA.CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)
_C.DATA.DATASET_TYPE = 'cifer10'
_C.DATA.IMG_SIZE = 32
_C.DATA.INPUT_DIM = 3
_C.DATA.NUM_WORKERS = 32
_C.DATA.SAMPLING_CLASS_COUNTS = [5000, 5000, 2500, 5000, 2500, 5000, 5000, 5000, 5000, 2500]
# _C.DATA.SAMPLING_CLASS_COUNTS = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
_C.DATA.TRAIN_CLASS_COUNTS = [4500, 4500, 2000, 4500, 2000, 4500, 4500, 4500, 4500, 2000]
# DataAug_List: ['random_rotation', 'random_horizontal_flip', 'random_vertical_flip', 'color_jitter', 'to_tensor', 'normalize']
_C.DATA.TRANSFORM_LIST = ['resize', 'to_tensor', 'random_horizontal_flip']
_C.DATA.VALIDATION_SIZE = 0.25
