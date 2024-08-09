import os
from os import listdir
from os.path import isfile, join
import sys
import subprocess
import opensmile
import speechpy
import numpy as np
import pandas as pd
from scipy import ndimage

def getPath(dataset_name):
    set = None
    if dataset_name == "trueness":
        audio_dir = "Full/"
        set = ["train", "test"]
    elif dataset_name == "cheese":
        audio_dir = "Full/"
        set = ["train", "test"]

    else:
        sys.exit("Error in the dataset name")

    #path = "/storage/raid1/homedirs/alice.delbosc/data/"+dataset_name+"_data/raw_data/Audio/"
    path = "/gpfsdswork/projects/rech/urk/uln35en/raw_data/trueness/Audio/"
    processed_dir = "processed/"

    return path, processed_dir, audio_dir, set

def createCsvFile(dir, out):
    print("*"*10, "createCsvFile", "*"*10)
    for f in listdir(dir):
        key = f[0:-4]
        if(".wav" in f and not isfile(join(out, key + ".csv"))):
            print("generation of", key)
            smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                    )
            result_smile = smile.process_file(join(dir,f))
            result_smile.to_csv(join(out, key + ".csv"), sep=',')


def removeOverlapAndAlignStep(out, objective_step):
    print("*"*10, "removeOverlapAndAlignStep", "*"*10)
    for csv_file in listdir(out + "origin/"):
        print(csv_file)
        df_audio = pd.read_csv(join(out + "origin/",csv_file))
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
        df_audio.to_csv(join(out,csv_file), index=False)

def addDerivative(out, audio_features):
    print("*"*10, "addDerivative", "*"*10)
    for csv_file in listdir(out):
        print(csv_file)
        df = pd.read_csv(join(out,csv_file))
        final_df = df.copy()
        for audio in audio_features:
            first = speechpy.processing.derivative_extraction(np.array(final_df[[audio]]), 1)
            second = speechpy.processing.derivative_extraction(first, 1)
            final_df["first_"+audio] = first
            final_df["second_"+audio] = second
        final_df.to_csv(join(out,csv_file), index=False)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    with_extract = sys.argv[2] ## if False, reuse the existing csv file for removeOverlapAndAlignStep and addDerivative

    audio_features = ["timestamp", 'Loudness_sma3','alphaRatio_sma3','hammarbergIndex_sma3','slope0-500_sma3','slope500-1500_sma3','spectralFlux_sma3','mfcc1_sma3',\
                      'mfcc2_sma3','mfcc3_sma3','mfcc4_sma3','F0semitoneFrom27.5Hz_sma3nz','jitterLocal_sma3nz','shimmerLocaldB_sma3nz','HNRdBACF_sma3nz',\
                        'logRelF0-H1-H2_sma3nz','logRelF0-H1-A3_sma3nz','F1frequency_sma3nz','F1bandwidth_sma3nz','F1amplitudeLogRelF0_sma3nz','F2frequency_sma3nz',\
                            'F2bandwidth_sma3nz','F2amplitudeLogRelF0_sma3nz','F3frequency_sma3nz','F3bandwidth_sma3nz','F3amplitudeLogRelF0_sma3nz']

    timestep = 0.04
    path, processed_dir, audio_dir, set = getPath(dataset_name)

    for set_name in set:
        dir = join(path, audio_dir, set_name)
        out = join(path, processed_dir, set_name)
        if(not os.path.exists(out)):
            os.mkdir(out)
        if(not os.path.exists(out + "origin/")):
            os.mkdir(out + "origin/")
        if(with_extract == "True"):
            createCsvFile(dir, out + "origin/")
        removeOverlapAndAlignStep(out, timestep)
        addDerivative(out, audio_features[1:])