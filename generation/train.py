import os
import constants.constants as constants
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.logger import MyLogger

class Train():
    def __init__(self):
        super(Train, self).__init__()
        checkpoint_callback = ModelCheckpoint(dirpath=constants.saved_path, every_n_epochs=constants.log_interval, save_top_k = -1, filename='{epoch}')
        logger = MyLogger()

        # Instantiate Profiler
        # profiler = PyTorchProfiler(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        #                             record_shapes=False,
        #                             profile_memory=True,
        #                             with_stack=False)

        constants.number_of_gpu = int(os.environ['SLURM_GPUS_ON_NODE']) * int(os.environ['SLURM_NNODES'])
        print("number of gpu", constants.number_of_gpu)
        trainer_args = {'accelerator': 'gpu', 
                # 'devices': int(os.environ['SLURM_GPUS_ON_NODE']), 
                # 'num_nodes': int(os.environ['SLURM_NNODES']),
                # 'strategy': 'ddp',
                "max_epochs": constants.n_epochs, 
                "check_val_every_n_epoch": 1, 
                "log_every_n_steps":10,
                "enable_progress_bar": False, 
                "callbacks": [checkpoint_callback],
                #"detect_anomaly":True,
                "logger": logger,
                #'profiler': profiler,
                #'sync_batchnorm': True
                }
        if(constants.do_resume):
            trainer_args["resume_from_checkpoint"] = constants.saved_path + "epoch="+str(constants.resume-1)+".ckpt"
        
        dm = constants.customDataModule()
        dm.prepare_data()
        dm.setup(stage="fit")
        # Marquez les données comme préparées pour éviter de les préparer à nouveau
        dm.is_prepared = True
        constants.train_model(trainer_args, dm)
        