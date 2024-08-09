from os import listdir
from os.path import isfile, join
import opensmile
import speechpy
import numpy as np
import pandas as pd
import pickle
import sys
import librosa
from transformers import Wav2Vec2Processor
import torch
from transformers import HubertModel
import argparse

visual_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]


def getPath():
    path = "/gpfsdswork/projects/rech/urk/uln35en/raw_data/audio_file/"
    wav_dir = "wav/"
    processed_dir = "processed/"
    anno_dir = "annotation/"
    complete_anno_dir = "complete_annotation/"
    set_dir = "final_data/"

    return path, wav_dir, processed_dir, anno_dir, complete_anno_dir, set_dir


def createCsvFile(wav_file, origin_processed_file):
    print("*"*10, "createCsvFile", "*"*10)
    smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
    result_smile = smile.process_file(wav_file)
    result_smile.to_csv(origin_processed_file, sep=',')

def removeOverlapAndAlignStep(origin_processed_file, processed_file, objective_step):
    print("*"*10, "removeOverlapAndAlignStep", "*"*10)
    df_audio = pd.read_csv(origin_processed_file)
    df_audio["start"] = df_audio["start"].transform(lambda x :  pd.Timedelta(x).total_seconds())
    df_audio["end"] = df_audio["end"].transform(lambda x :  pd.Timedelta(x).total_seconds())
    #we remove the overlaps and recalculate with averages
    df_audio_wt_overlap = df_audio.copy()
    df_audio_wt_overlap = df_audio_wt_overlap.rename(columns={"start":"timestamp"})
    df_audio_wt_overlap = df_audio_wt_overlap.drop(columns=["end"])
    df_audio_wt_overlap = df_audio_wt_overlap[audio_features]
    #mean the value of overlap
    for index, row in df_audio_wt_overlap.iterrows():
        if index != 0:
            df_audio_wt_overlap.at[index, "Loudness_sma3"] = (row["Loudness_sma3"] + df_audio.iloc[[index-1]]["Loudness_sma3"]) / 2

    #we change the timestep to match the openface timestep
    df_audio_wt_overlap["timestamp"] = df_audio_wt_overlap["timestamp"].astype(float)
    df_audio_wt_overlap["timestamp"] = ((df_audio_wt_overlap["timestamp"]/objective_step).astype(int)) * objective_step
    df_audio_wt_overlap["timestamp"] = round(df_audio_wt_overlap["timestamp"],2)
    df_audio_wt_overlap = df_audio_wt_overlap.groupby(by=["timestamp"]).mean()
    df_audio_wt_overlap = df_audio_wt_overlap.reset_index()
    df_audio = df_audio_wt_overlap
    df_audio.to_csv(processed_file, index=False)

def create_complete_ipu_file(anno_file):
    df = pd.read_csv(anno_file, names=["ipu", "begin", "end", "speak"])
    df["bool_speak"] = df["speak"].transform(lambda x : 0 if x == "#" else 1)
    df = df.drop(columns=['ipu'])
    return df

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

#pour chaque IPU extrait par sppas (IPU1, IPU2...) je veux ajouter direct les labels choisis à la main (donc pas besoin de faire l'extraction gpt)
def create_set(file_name, output_file, wav_file, wav_processed_file, df_ipu, gender, segment_length, overlap, timestep):
    print("process of", file_name)

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
        
    final_dict = {"wav_array": [], "hubert_array": [], "hubert_french_array":[], 
                    "time_array": [], "details_time": [], 
                    "behaviour_array": [], "previous_behaviour": [], "seq_previous_behaviour": [], "final_behaviour": [],
                    "prosody_array":[], 
                    "speak_or_not": []}

    df_ipu_align = change_timestep(df_ipu, timestep)
    df_ipu_align.timestamp = df_ipu_align.timestamp.astype(float)
    end_time_annotations = df_ipu_align["timestamp"].iloc[-1]

    df_prosody = pd.read_csv(wav_processed_file)
    prosody_features = df_prosody.columns.drop("timestamp")
    end_time_prosody = df_prosody["timestamp"].iloc[-1]

    df_result = df_prosody.merge(df_ipu_align, on='timestamp', how='inner')
    df_result.drop(["begin", "end"], axis=1, inplace=True)
    end_time_result = df_result["timestamp"].iloc[-1]

    print("End time:", "[result:", end_time_result, "]", "[prosody:", end_time_prosody, "]", "[ipu:", end_time_annotations, "]")
    del df_ipu, df_ipu_align, df_prosody
    final_dict["gender"] = gender
    final_dict["key"] = file_name
    #cut into segment of length "segment_length" with overlap, and create the array of features
    t1, t2 = 0, segment_length
    while t2 <= end_time_result:
        print(t1,t2)
        #speech (wavtovec)
        speech_array, sampling_rate = librosa.load(wav_file, offset=t1, duration=t2-t1, sr=16000)
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

        #time
        final_dict["time_array"].append((t1,t2))
        final_dict["details_time"].append(second_cut["timestamp"].values)

        t1, t2 = round(t1 + segment_length - overlap,2), round(t2 + segment_length - overlap,2)

    with open(output_file, 'wb') as f:
        pickle.dump(final_dict, f)
    del final_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', help='file name', default='')
    parser.add_argument('-segment', type=int, default=2)
    parser.add_argument('-overlap', type=float, default=0)

    parser.add_argument('-gender', default="F")
    args = parser.parse_args()


    file_name = args.file
    segment_length = args.segment #secondes
    timestep = 0.04
    overlap = args.overlap

    audio_features = ["timestamp", 'Loudness_sma3','alphaRatio_sma3','hammarbergIndex_sma3','slope0-500_sma3','slope500-1500_sma3','spectralFlux_sma3','mfcc1_sma3',\
                      'mfcc2_sma3','mfcc3_sma3','mfcc4_sma3','F0semitoneFrom27.5Hz_sma3nz','jitterLocal_sma3nz','shimmerLocaldB_sma3nz','HNRdBACF_sma3nz',\
                        'logRelF0-H1-H2_sma3nz','logRelF0-H1-A3_sma3nz','F1frequency_sma3nz','F1bandwidth_sma3nz','F1amplitudeLogRelF0_sma3nz','F2frequency_sma3nz',\
                            'F2bandwidth_sma3nz','F2amplitudeLogRelF0_sma3nz','F3frequency_sma3nz','F3bandwidth_sma3nz','F3amplitudeLogRelF0_sma3nz']
    
    path, wav_dir, processed_dir, anno_dir, complete_anno_dir, set_dir = getPath()

    wav_file = join(path, wav_dir, file_name+".wav")
    wav_processed_file = join(path, processed_dir, file_name+".csv")
    wav_origin_processed_file = join(path, processed_dir, "origin", file_name+".csv")
    anno_file = join(path, anno_dir, file_name+".csv")
    set_file = join(path, set_dir, str(segment_length), file_name+".p")
    print(set_file)

    createCsvFile(wav_file, wav_origin_processed_file)
    removeOverlapAndAlignStep(wav_origin_processed_file, wav_processed_file, timestep)
    df_anno = create_complete_ipu_file(anno_file)
    print(df_anno)
    create_set(file_name, set_file, wav_file, wav_processed_file, df_anno, args.gender, segment_length, overlap, timestep)
