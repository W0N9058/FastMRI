from easydict import EasyDict as edict

class Config:
    # dataset
    DATASET = edict()
    DATASET.TYPE = 'FastMRI'
    DATASET.TRAIN_PATH = '/home/Data/train'
    DATASET.VAL_PATH = '/home/Data/val'
    DATASET.SAMPLE_RATE = 1.0  # 샘플링 비율 추가
    DATASET.INPUT_KEY = 'kspace'  # input_key 추가
    DATASET.TARGET_KEY = 'target'  # target_key 추가
    DATASET.MAX_KEY = 'max'  # max_key 추가
    DATASET.SCALE = 1  # SCALE 추가

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 8
    DATALOADER.NUM_WORKERS = 4

    # model
    MODEL = edict()
    MODEL.D = edict()
    MODEL.D.IN_CHANNEL = 1  # FastMRI 데이터셋의 경우 1채널로 변경
    MODEL.D.N_CHANNEL = 32  # 필요한 N_CHANNEL 추가
    MODEL.BBL_WEIGHT = 0.5  # BBL_WEIGHT 추가

    # solver
    SOLVER = edict()
    # discriminator
    SOLVER.D_OPTIMIZER = 'Adam'
    SOLVER.D_BASE_LR = 1e-4
    SOLVER.D_BETA1 = 0.9
    SOLVER.D_BETA2 = 0.999
    SOLVER.D_WEIGHT_DECAY = 0
    SOLVER.D_MOMENTUM = 0
    SOLVER.D_STEP_ITER = 1
    # generator
    SOLVER.G_OPTIMIZER = 'Adam'  # Generator optimizer 추가
    SOLVER.G_BASE_LR = 1e-4  # Generator learning rate 추가
    SOLVER.G_BETA1 = 0.9  # Generator beta1 추가
    SOLVER.G_BETA2 = 0.999  # Generator beta2 추가
    SOLVER.G_WEIGHT_DECAY = 0  # Generator weight decay 추가
    SOLVER.G_MOMENTUM = 0  # Generator momentum 추가
    # both G and D
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.T_PERIOD = [200000, 400000, 600000]
    SOLVER.MAX_ITER = SOLVER.T_PERIOD[-1]

    # initialization
    CONTINUE_ITER = None

    # log and save
    LOG_PERIOD = 20
    SAVE_PERIOD = 10000

    # validation
    VAL = edict()
    VAL.PERIOD = 10000
    VAL.TYPE = 'FastMRI'
    VAL.DATASETS = ['FastMRI']
    VAL.SPLITS = ['VAL']
    VAL.PHASE = 'val'
    VAL.INPUT_HEIGHT = None
    VAL.INPUT_WIDTH = None
    VAL.SCALE = DATASET.SCALE
    VAL.REPEAT = 1
    VAL.VALUE_RANGE = 255.0
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.SAVE_IMG = False
    VAL.TO_Y = True
    VAL.CROP_BORDER = VAL.SCALE

config = Config()