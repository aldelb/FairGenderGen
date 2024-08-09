import argparse
from os.path import join, isfile
from os import listdir
import sys
import time
import pandas as pd
import numpy as np
import pickle
import librosa
from transformers import Wav2Vec2Processor
import torch
from transformers import HubertModel


CLUSTER="jean-zay"
visual_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]


def change_timestep(df, objective_step, begin_column="begin", end_column="end"):
    current_time = 0
    last_index = len(df)-1
    new_df = pd.DataFrame([], columns=df.columns)
    index = 0
    while(index <= last_index):
        while(current_time + objective_step <= df.at[index, end_column]):
            new_row = df.loc[index].copy()
            new_row['timestamp'] = current_time
            new_df = pd.concat([new_df, new_row.to_frame().T], ignore_index=True)
            current_time = round(current_time + objective_step,2)
        index = index + 1
    return new_df

def create_hubert_embedding(wav_to_vec_vector, audio_encoder):
        audio_dim = audio_encoder.encoder.config.hidden_size
        embedding = audio_encoder(wav_to_vec_vector, output_hidden_states=True)
        correct_embedding = []
        for i in range(len(embedding.hidden_states)):
            correct_embedding.append(torch.cat([embedding.hidden_states[i], embedding.hidden_states[i][:, -1:, :]], dim=1))
        return correct_embedding
    
def create_set(regenerate_flag, init_output_path, output_path, audio_path, visual_path, ipu_path, prosody_path, moveToZero, segment_length, overlap_train, overlap_test, timestep, data_details):
    modelname = "/gpfsdswork/projects/rech/urk/uln35en/model/hubert-large-ls960-ft/"
    french_modelname = "/gpfsdswork/projects/rech/urk/uln35en/model/exp_w2v2t_fr_hubert_s767/"

    processor = Wav2Vec2Processor.from_pretrained(modelname)
    audio_encoder = HubertModel.from_pretrained(modelname)
    audio_encoder.feature_extractor._freeze_parameters()
    for name, param in audio_encoder.named_parameters():
        param.requires_grad = False

    french_processor = Wav2Vec2Processor.from_pretrained(french_modelname)
    french_audio_encoder = HubertModel.from_pretrained(french_modelname)
    french_audio_encoder.feature_extractor._freeze_parameters()
    for name, param in french_audio_encoder.named_parameters():
        param.requires_grad = False
    
    for file in listdir(visual_path):
        if(".csv" not in file):
            continue  
        
        key = file[0:-4]
        print("process of", key)

        if(data_details[key]["set"] == "train"):
            overlap = overlap_train
        else:
            overlap = overlap_test

        final_dict = {"key": key, "wav_path": join(audio_path, key + ".wav"), "behaviour_path": join(visual_path, file), "prosody_path": join(prosody_path, file), "ipu_path": join(ipu_path, key+".xlsx"),
                        "wav_array": [], "hubert_array": [], "hubert_french_array":[], 
                        "time_array": [], "details_time": [], 
                        "behaviour_array": [], "previous_behaviour": [], "seq_previous_behaviour": [], "final_behaviour": [],
                        "prosody_array":[], 
                        "speak_or_not": []}
        final_path  = join(output_path,  key + ".p")
        if(not isfile(final_path) or regenerate_flag):
            df_ipu = pd.read_excel(final_dict["ipu_path"])[["begin", "end", "speak", "bool_speak"]]
            df_ipu_align_path = join(ipu_path, "align", file)
            if(not isfile(df_ipu_align_path)):
                df_ipu_align = change_timestep(df_ipu, timestep)
                df_ipu_align.to_csv(df_ipu_align_path, index=False)
            else:
                df_ipu_align = pd.read_csv(df_ipu_align_path)
            df_ipu_align.timestamp = df_ipu_align.timestamp.astype(float)
            end_time_annotations = df_ipu_align["timestamp"].iloc[-1]

            selected_video_features = ["timestamp"] + visual_features
            df_video = pd.read_csv(final_dict["behaviour_path"])[selected_video_features]
            previous_behaviour = df_video[visual_features].iloc[0]
            previous_behaviour[visual_features] = 0
            seq_previous_behaviour = df_video[visual_features].iloc[[0,1]]
            seq_previous_behaviour[visual_features] = 0

            end_time_video = df_video["timestamp"].iloc[-1]

            df_prosody = pd.read_csv(final_dict["prosody_path"])
            prosody_features = df_prosody.columns.drop("timestamp")
            end_time_prosody = df_prosody["timestamp"].iloc[-1]
            df_result_path = join(init_output_path, key + ".csv")
            if(not isfile(df_result_path) or regenerate_flag):
                df_result = df_video.merge(df_prosody, on='timestamp', how='inner')
                df_result = df_result.merge(df_ipu_align, on='timestamp', how='inner')
                df_result.drop(["begin", "end"], axis=1, inplace=True)
                df_result.to_csv(df_result_path, index=False)
            else:
                df_result = pd.read_csv(df_result_path)
            
            final_dict["final_behaviour_init"] = df_result[visual_features]
            if moveToZero:
                index_listening_behavior = df_result.where(df_result['bool_speak'] == 0).dropna().index
                df_result.loc[index_listening_behavior, visual_features] = 0
            end_time_result = df_result["timestamp"].iloc[-1]

            print("End time:", "[result:", end_time_result, "]", "[prosody:", end_time_prosody, "]", "[ipu:", end_time_annotations, "]", "[video:", end_time_video, "]")
            del df_video, df_ipu, df_ipu_align, df_prosody

            final_dict["final_behaviour"] = df_result[visual_features]
            #cut into segment of length "segment_length" with overlap, and create the array of features
            t1, t2 = 0, segment_length

            final_dict["gender"] = data_details[key]["genre"]
            final_dict["role"] = data_details[key]["role"]
            final_dict["set"] = data_details[key]["set"]
            final_dict["attitude"] = data_details[key]["attitude_harceuleur"]
            
            while t2 <= end_time_result:
                #speech (wavtovec)
                speech_array, sampling_rate = librosa.load(final_dict["wav_path"], offset=t1, duration=t2-t1, sr=16000)
                input_values = processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
                final_dict["hubert_array"].append(create_hubert_embedding(input_values, audio_encoder))

                # french_input_values = french_processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
                # final_dict["hubert_french_array"].append(create_hubert_embedding(french_input_values, french_audio_encoder))

                first_cut = df_result[df_result["timestamp"] < t2]
                second_cut = first_cut[first_cut["timestamp"] >= t1]

                #speak_or_not
                final_dict["speak_or_not"].append(second_cut["bool_speak"].values)

                #speech features (opensmile)
                final_dict["prosody_array"].append(second_cut[prosody_features].values)

                #behaviour (openface)
                final_dict["behaviour_array"].append(second_cut[visual_features].values)

                #previous behaviour
                final_dict["previous_behaviour"].append(previous_behaviour.values)
                previous_behaviour = second_cut[visual_features].iloc[-1]

                final_dict["seq_previous_behaviour"].append(seq_previous_behaviour.values)
                seq_previous_behaviour = second_cut[visual_features].iloc[[-2,-1]]

                #time
                final_dict["time_array"].append((t1,t2))
                final_dict["details_time"].append(second_cut["timestamp"].values)

                t1, t2 = round(t1 + segment_length - overlap,2), round(t2 + segment_length - overlap,2)

            with open(final_path, 'wb') as f:
                pickle.dump(final_dict, f)
            del final_dict


def getPath(dataset_name, moveToZero, segment_length):
    if(CLUSTER=="jean-zay"):
        general_path = "/gpfsdswork/projects/rech/urk/uln35en/"
        dataset_path = general_path + "raw_data/"+dataset_name+"/"
        init_output_path = dataset_path + "/final_data/"

        if(moveToZero):
            output_path = join(init_output_path, str(segment_length), "moveSpeakerOnly")
        else:
            output_path = join(init_output_path, str(segment_length), "none")
        
        audio_path = dataset_path + "audio/full/"
        visual_path = dataset_path + "video/processed/" 
        ipu_path = dataset_path + "annotation/processed/ipu_with_tag/"
        prosody_path = dataset_path + "audio/processed/"
        details_file = dataset_path + "details.xlsx"
        details_df = pd.read_excel(details_file)
        data_details = details_df.set_index("nom").to_dict(orient='index')
    else:
        sys.exit("Error in the cluster name")
    return init_output_path, output_path, audio_path, visual_path, ipu_path, prosody_path, data_details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset')
    parser.add_argument('-zeroMove', action='store_true')
    parser.add_argument('-segment', type=int, default=2)
    args = parser.parse_args()

    dataset_name = args.dataset
    moveToZero = args.zeroMove
    segment_length = args.segment #secondes
    
    timestep = 0.04
    regenerate_flag = True
    


    overlap_train = round(0.1 * segment_length,2) 
    overlap_test = round(0.1 * segment_length,2) 
    print("length for train", segment_length, "overlap for train:", overlap_train, "overlap for test", overlap_test)

    init_output_path, output_path, audio_path, visual_path, ipu_path, prosody_path, data_details = getPath(dataset_name, moveToZero, segment_length)
    create_set(regenerate_flag, init_output_path, output_path, audio_path, visual_path, ipu_path, prosody_path, moveToZero, segment_length, overlap_train, overlap_test, timestep, data_details)
    print("*"*10, "end of creation", "*"*10)

    return 0

if __name__ == "__main__":
    sys.exit(main())
