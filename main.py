# Configs
from omegaconf import OmegaConf, DictConfig
import os
import hydra
import json

# Pytorch Lightning
import pytorch_lightning as pl
from lightning.pytorch.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    # PyTL has various utility function like seed_everything which
    # Seed random number generators in : PyTorch, Numpy, Pytho.random and env var PL_GLOBAL_SEED
    pl.seed_everything(config.seed)

    # Integrating with Neptune
    logger = True
    if config.use_neptune:
        logger = NeptuneLogger(**config.neptune)
        logger.log_hyperparams(config)

    # Instantiate model and data
    data = hydra.utils.instantiate({**config.dataset, **config.dataloader})
    data_shape, label_shape = data.get_shape()
    model = hydra.utils.instantiate({**config.model, **config.optimizer}, shape=data_shape)
    
    # Train the model
    trainer = pl.Trainer(logger=logger, **config.trainer)
    trainer.fit(model, data)
    
    # Testing
    trainer.test(model, data)

if __name__ == '__main__':
    main()