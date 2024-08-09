import os
import constants.constants as constants
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.logger import MyLogger

class Generate():
    def __init__(self, datasets, epoch, file=None):
        super(Generate, self).__init__()
        if(datasets != ""):
            constants.datasets = datasets.split(",")

        checkpoint_callback = ModelCheckpoint(dirpath=constants.saved_path, every_n_epochs=constants.log_interval, save_top_k = -1, filename='{epoch}')
        logger = MyLogger()

        constants.number_of_gpu = int(os.environ['SLURM_GPUS_ON_NODE']) * int(os.environ['SLURM_NNODES'])
        trainer_args = {'accelerator': 'gpu', 
                "max_epochs": constants.n_epochs, 
                "check_val_every_n_epoch": 1, 
                "log_every_n_steps":10,
                "enable_progress_bar": False, 
                "callbacks": [checkpoint_callback],
                "logger": logger}

        if(file!=None):
            self.dm = constants.customDataModule(fake_examples=False, predict=True, one_file=True, file=file)
            self.dm.prepare_data()
            self.dm.setup(stage="predict_one_file")
        else:
            self.dm = constants.customDataModule(fake_examples=False, predict=True)
            self.dm.prepare_data()
            self.dm.setup(stage="predict")
        
        constants.generate_motion(epoch, trainer_args, self.dm, file)