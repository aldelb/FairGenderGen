import pandas as pd
import pytorch_lightning as pl
import torch
import os
from os.path import join, isdir, isfile
import pickle
import numpy as np
import constants.constants as constants
from utils.data_utils import scale_from_scratch, scale
from utils.labels import label_to_one_hot, get_maj_label
from torch.utils.data import DataLoader, Dataset



class CustomDataset(Dataset):
    def __init__(self, X_audio, Y, speak_or_not, gender=None, interval=None, details_time=None, final_Y=None, predict=False, keys=None):
        self.X_audio = X_audio
        self.Y = Y
        self.speak_or_not = speak_or_not
        self.gender = gender
        self.interval = interval
        self.details_time = details_time
        self.final_Y = final_Y
        self.predict = predict
        self.keys = keys

    def __len__(self):
        return len(self.X_audio)

    def __getitem__(self, i):
        if(self.predict):
            return self.X_audio[i], self.details_time[i], self.keys[i], self.gender[i]
        return self.X_audio[i], self.Y[i], self.gender[i]
    
    def get_final_test_videos(self):
        return self.keys, self.final_Y
            

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, fake_examples = None, predict = None, one_file=None, file=None):
        super(CustomDataModule, self).__init__()
        self.is_prepared = False
        self.batch_size = constants.batch_size
        if(fake_examples == None):
            self.fake_examples = constants.fake_target
        else:
            self.fake_examples = fake_examples
        
        if(predict == None):
            self.predict = False
        else:
            self.predict = True

        if(one_file == None):
            self.one_file = False
        else:
            self.one_file = True
            self.file = file

    
    def load_data(self, datasets, datasets_properties, set_type, path):
        dataset = datasets[0]
        final_path = join(path, dataset, "final_data/", datasets_properties)
        details_file = constants.data_path + dataset + "/details.xlsx"
        details = pd.read_excel(details_file)
        list_of_keys = details[["nom", "set"]].where(details["set"] == set_type).dropna()["nom"].values

        X_audio, Y_behaviour, details_time, speak_or_not, gender, final_Y, keys, raw_keys = ([] for _ in range(8))

        for key in list_of_keys:
            key_file = join(final_path, key + ".p")
            if(isfile(key_file)):
                with open(key_file, 'rb') as f:
                    data = pickle.load(f)
                if(constants.french_hubert):
                    for value_hubert_seq in data["hubert_french_array"]:
                        X_audio.extend(value_hubert_seq[constants.hidden_state_index]) 
                else:
                    for value_hubert_seq in data["hubert_array"]:
                        X_audio.extend(value_hubert_seq[constants.hidden_state_index]) 
                Y_behaviour.extend(data["behaviour_array"])
                details_time.extend(data["details_time"])
                speak_or_not.extend(data["speak_or_not"])
                gender.extend([ele for ele in [data["gender"]] for _ in range(len(data["behaviour_array"]))])
                final_Y.append(data["final_behaviour"])
                keys.extend([ele for ele in [data["key"]] for _ in range(len(data["behaviour_array"]))])
                raw_keys.append(data["key"])
                # Free up memory occupied by the 'data' variable
                del data

        speak_or_not = torch.as_tensor(list(map(list, speak_or_not))).unsqueeze(-1)
        speak_or_not = torch.repeat_interleave(speak_or_not, 2, dim=1)

        X_audio = torch.stack(X_audio)
        print(X_audio.shape)
        X_audio_final = torch.cat((speak_or_not, X_audio), dim=2)

        gender = [gender[i] if value != "silence" else value for i, value in enumerate(keys)]
        one_hot_tensor_gender = torch.stack([label_to_one_hot(label, "gender") for label in gender])        
        Y_behaviour = torch.as_tensor(np.array(Y_behaviour))
        
        constants.seq_len = Y_behaviour.shape[1]
        constants.audio_dim = X_audio_final.shape[2]

        return X_audio_final, Y_behaviour, speak_or_not, one_hot_tensor_gender, details_time, final_Y, keys, raw_keys

    def load_file_data(self, filename):
        X_audio = []
        details_time = []
        speak_or_not = []
        gender = []
        keys = []
        raw_keys = []

        path_to_single_file = constants.data_path + "/audio_file/final_data/4/"

        with open(join(path_to_single_file, filename+".p"), 'rb') as f:
            data = pickle.load(f)
            if(constants.french_hubert):
                for value_hubert_seq in data["hubert_french_array"]:
                    X_audio.extend(value_hubert_seq[constants.hidden_state_index]) 
            else:
                for value_hubert_seq in data["hubert_array"]:
                    X_audio.extend(value_hubert_seq[constants.hidden_state_index]) 
            details_time.extend(data["details_time"])
            speak_or_not.extend(data["speak_or_not"])
            gender.extend([ele for ele in [data["gender"]] for _ in range(len(data["details_time"]))])
            keys.extend([ele for ele in [data["key"]] for _ in range(len(data["details_time"]))])
            raw_keys.append(data["key"])

        speak_or_not = torch.as_tensor(list(map(list, speak_or_not))).unsqueeze(-1)
        speak_or_not = torch.repeat_interleave(speak_or_not, 2, dim=1)

        X_audio = torch.stack(X_audio)
        X_audio_final = torch.cat((speak_or_not, X_audio), dim=2)

        gender = [gender[i] if value != "silence" else value for i, value in enumerate(keys)]
        one_hot_tensor_gender = torch.stack([label_to_one_hot(label, "gender") for label in gender])
        constants.audio_dim = X_audio_final.shape[2]
        constants.seq_len = len(details_time[0])
        print(constants.audio_dim, constants.seq_len)
        return X_audio_final, speak_or_not, one_hot_tensor_gender, details_time, keys, raw_keys


    def prepare_data(self):
        if not self.is_prepared:
            print("Lauching of prepare_data")   
            path = constants.data_path
            datasets = constants.datasets
            datasets_properties = constants.datasets_properties
            if not self.predict:
                self.X_train_audio, self.Y_train, self.speak_or_not_train, self.gender_train, self.details_time_train, _, _, _ = self.load_data(datasets, datasets_properties, "train", path)
            
            if not self.one_file:
                self.X_test_audio, self.Y_test, self.speak_or_not_test, self.gender_test, self.details_time_test, self.final_Y, self.keys_test, self.raw_keys = self.load_data(datasets, datasets_properties, "test", path)
            else:
                self.X_test_audio, self.speak_or_not_test, self.gender_test, self.details_time_test, self.keys_test, self.raw_keys = self.load_file_data(self.file)



    def setup(self, stage=None):
        print("Lauching of setup - ", stage)
        if not self.is_prepared:   
            dir_scaler = join(constants.saved_path, "scaler")
            os.makedirs(dir_scaler, exist_ok=True)
            if stage == 'fit':
                self.Y_scaled_train, self.y_scaler = scale_from_scratch(self.Y_train, "tanh")
                self.train_dataset = CustomDataset(X_audio=self.X_train_audio, Y=self.Y_scaled_train, speak_or_not=self.speak_or_not_train, gender=self.gender_train)

                self.Y_scaled_test = scale(self.Y_test, self.y_scaler)
                self.test_dataset = CustomDataset(X_audio=self.X_test_audio, Y=self.Y_scaled_test, speak_or_not=self.speak_or_not_test, gender=self.gender_test)

                pickle.dump(self.y_scaler, open(join(dir_scaler, 'scaler_y.pkl'), 'wb'))

                if(self.fake_examples):
                    self.create_fake_examples(self.X_train_audio, self.Y_scaled_train, self.speak_or_not_train, self.gender_train)

            elif stage == "predict":
                self.y_scaler = pickle.load(open(join(dir_scaler, 'scaler_y.pkl'), 'rb'))
                self.Y_scaled_test = scale(self.Y_test, self.y_scaler)
                self.test_dataset = CustomDataset(X_audio=self.X_test_audio, Y=None, speak_or_not=self.speak_or_not_test, gender=self.gender_test, details_time=self.details_time_test, keys=self.keys_test, predict=True)

            elif stage == "predict_one_file":
                self.y_scaler = pickle.load(open(join(dir_scaler, 'scaler_y.pkl'), 'rb'))
                self.test_dataset = CustomDataset(X_audio=self.X_test_audio, Y=None, speak_or_not=self.speak_or_not_test, gender=self.gender_test, details_time=self.details_time_test, keys=self.keys_test, predict=True)
            
            elif stage == "evaluate":
                self.test_dataset = CustomDataset(X_audio=None, Y=None, speak_or_not=None, gender=None, labels=None, details_time=None, keys=self.raw_keys, final_Y=self.final_Y, predict=False)
            
            self.is_prepared = True



    def create_fake_examples(self, X_audio, Y, speak_or_not, gender):
        print("Lauching of create_fake_examples")
        speak_x_audio = []
        speak_y = []
        speak_labels = {"gender" : []}
        no_speak_x_audio = []
        no_speak_y = []
        no_speak_labels = {"gender" : []}
        
        for index, speak_boolean_value in enumerate(speak_or_not):
            speak_boolean_value = speak_boolean_value.to(dtype=torch.float32)
            if(torch.mean(speak_boolean_value).item() < 0.2): #if the person is mostly speaking
                speak_x_audio.append(X_audio[index])
                speak_y.append(Y[index])
                speak_labels["gender"].append(gender[index])
            elif(torch.mean(speak_boolean_value) > 0.8):
                no_speak_x_audio.append(X_audio[index])
                no_speak_y.append(Y[index])
                no_speak_labels["gender"].append(gender[index])

        self.speak_examples = (torch.stack(speak_x_audio, 0).squeeze(), torch.stack(speak_y, 0).squeeze(), torch.stack(speak_labels["gender"],0).squeeze())
        self.no_speak_examples = (torch.stack(no_speak_x_audio, 0).squeeze(), torch.stack(no_speak_y, 0).squeeze(), torch.stack(no_speak_labels["gender"],0).squeeze())


    def train_dataloader(self):
        print("train dataloader")
        return DataLoader(self.train_dataset, batch_size=int(self.batch_size/constants.number_of_gpu), shuffle=True, num_workers=4)

    def val_dataloader(self):
        print("val dataloader")
        return DataLoader(self.test_dataset, batch_size=int(self.batch_size/constants.number_of_gpu), shuffle=False, num_workers=4)

    def predict_dataloader(self):
        print("predict dataloader")
        return DataLoader(self.test_dataset, batch_size=int(self.batch_size/constants.number_of_gpu), shuffle=False, num_workers=4)

