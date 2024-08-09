import pandas as pd

def createFinalFile(path_data_out, key, df_list, file=None):
    df = []
    for new_df in df_list:
        if(len(df) == 0):
            df = new_df
        else:
            df = pd.concat([df, new_df], ignore_index=True)
       
    if(len(df) > 0):
        print("len df before mean", len(df))
        df = df.groupby('timestamp').mean().reset_index()
        print("len df after mean", len(df))
        df.set_index("timestamp", inplace =True)
        if(file!=None):
            save_file = path_data_out + file + ".csv"
        else:
            save_file = path_data_out + key + ".csv"
        df.to_csv(save_file)