import os
import hydra
import pytorch_lightning as pl
from importlib import import_module
from minsu3d.data.joint_data_module import JointDataModuleT
from minsu3d.data.joint_data_module import JointDataModuleV
from minsu3d.model.joint_preprocess import JointPreprocessModelT
from minsu3d.model.joint_preprocess import JointPreprocessModelV


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)

    print("==> initializing data ...")
    train_data_module = JointDataModuleT(cfg)
    val_data_module = JointDataModuleV(cfg)

    print("==> initializing trainer ...")
    trainer_t = pl.Trainer(callbacks=False, logger=False, **cfg.model.trainer)
    trainer_v = pl.Trainer(callbacks=False, logger=False, **cfg.model.trainer)

    print("==> initializing model ...")
    process_train = JointPreprocessModelT(cfg)
    process_val = JointPreprocessModelV(cfg)

    print("==> start generating joint outputs ...")
    trainer_t.fit(model=process_train, datamodule=train_data_module)
    trainer_v.fit(model=process_val, datamodule=val_data_module)


if __name__ == '__main__':
    main()
