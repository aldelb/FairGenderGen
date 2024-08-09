import os
from os.path import isdir
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import constants.constants as constants
from utils.create_final_file import createFinalFile

class GenerateModel():
    def __init__(self):
        super(GenerateModel, self).__init__()


    def generate_motion(self, epoch, trainer_args, dm, file=None):
        path_data_out = constants.output_path + constants.model_path + "/epoch_" + epoch + "/"
        if(not isdir(path_data_out)):
            os.makedirs(path_data_out, exist_ok=True)
        checkpoint_epoch = int(epoch) - 1

        model = constants.model.load_from_checkpoint(constants.saved_path + "epoch="+str(checkpoint_epoch)+".ckpt")
        model.pose_scaler=dm.y_scaler

        trainer = pl.Trainer(**trainer_args)
        predictions = trainer.predict(model, dm)

        current_key = ""
        df_list = []
        for keys, preds, details_times, _, _, _, _, _, _, _ in predictions:
            for index, key in enumerate(keys):
                pred = preds[index]
                details_time = details_times[index]
                if(current_key != key): #process of a new video
                    if(current_key != ""):
                        createFinalFile(path_data_out, current_key, df_list)
                    print("*"*10, "Generation of video", key, "...", "*"*10)
                    current_key = key
                    df_list = []

                out = np.concatenate((np.array(details_time).reshape(-1,1), pred[:,:constants.eye_size], np.zeros((pred.shape[0], 3)), pred[:,constants.eye_size:]), axis=1)
                df = pd.DataFrame(data = out, columns = constants.openface_columns)
                df_list.append(df)
        createFinalFile(path_data_out, current_key, df_list, file)
        print("*"*10, "end of generation", "*"*10)
