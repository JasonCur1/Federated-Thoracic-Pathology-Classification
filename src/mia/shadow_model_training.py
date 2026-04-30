import torch
from mia import mia_utils
from pathlib import Path
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from baseline.config import Config, CFG
from baseline.model import ChestXrayClassifier
from baseline.train import build_callbacks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s – %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

SHADOW_PATH = CFG.project_root / 'shadow_data_small'

def load_data(cfg):
    logger.info('Loading shadow datamodule...')
    dm = mia_utils.load_data(cfg)
    logger.info('Data loaded successfully.')
    return dm

def build_model(dm, cfg):
    logger.info('Initializing shadow model...')
    model = ChestXrayClassifier(pos_weight=dm.pos_weight, cfg=cfg)
    logger.info('Model initialized.')
    return model

def build_trainer(cfg):
    logger.info('Setting up trainer...')

    tb_logger = TensorBoardLogger(
        save_dir=str(cfg.log_dir),
        name='shadow_model',
        default_hp_metric=False,
    )

    n_gpus = torch.cuda.device_count()
    strategy = 'ddp' if n_gpus > 1 else 'auto'

    if n_gpus <= 1 and cfg.devices == -1:
        effective_devices = max(n_gpus, 1)
    else:
        effective_devices = cfg.devices

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=effective_devices,
        strategy=strategy,
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        log_every_n_steps=cfg.log_every_n_steps,
        deterministic=False,
        logger=tb_logger,
        callbacks=build_callbacks(cfg),
        sync_batchnorm=True,
        use_distributed_sampler=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
    )

    logger.info('Trainer ready.')
    return trainer

def main():
    shadow_cfg = Config(data_root_path=SHADOW_PATH)
    shadow_cfg.experiment_name = 'shadow_model'
    shadow_cfg.checkpoint_dir_path = Path(shadow_cfg.project_root) / 'checkpoints'
    
    pl.seed_everything(shadow_cfg.seed, workers=True)

    shadow_dm = load_data(shadow_cfg)
    
    shadow_model = build_model(shadow_dm, shadow_cfg)
    
    trainer = build_trainer(shadow_cfg)
    
    logger.info('Starting training...')
    trainer.fit(shadow_model, datamodule=shadow_dm)
    logger.info('Training complete.')

    

if __name__ == '__main__':
    main()