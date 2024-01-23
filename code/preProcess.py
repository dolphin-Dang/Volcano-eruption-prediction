import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data_path = 'data/train/'
label_path = 'data/train.csv'

def process_one_file(seg):
    '''
    Return a features pd.Series() with 8 features of each sensor, total 80.
    '''
    df = pd.read_csv('data/train/' + str(seg) + '.csv')
    features = pd.Series()

    for i, column in enumerate(df.columns, start=1):
        if df[column].isnull().all():
            # all nan
            features['max_'+str(i)] = np.nan
            features['min_'+str(i)] = np.nan
            features['mean_'+str(i)] = np.nan
            features['std_'+str(i)] = np.nan
            features['median_'+str(i)] = np.nan
            features['kurtosis_'+str(i)] = np.nan
            features['skew_'+str(i)] = np.nan
            features['peak_freq_'+str(i)] = np.nan
        else:
            # not all nan
            df[column].fillna(df[column].median(), inplace=True) # fillna anyway
            features['max_'+str(i)] = df[column].max()
            features['min_'+str(i)] = df[column].min()
            features['mean_'+str(i)] = df[column].mean()
            features['std_'+str(i)] = df[column].std()
            features['median_'+str(i)] = df[column].median()
            features['kurtosis_'+str(i)] = kurtosis(df[column])
            features['skew_'+str(i)] = skew(df[column])
            peaks, _ = find_peaks(df[column])
            features['peak_freq_'+str(i)] = peaks[df[column][peaks].argmax()]

    return features

def deal_data():
    path = 'data/merged_data.csv'
    df = pd.read_csv(path)
    
    median_columns = [col for col in df.columns if col.startswith("median")]
    df.drop(columns=median_columns, inplace=True)
    
    df.fillna(df.median(), inplace=True)

    processed_path = 'data/merged_data_.csv'
    df.to_csv(processed_path, index=False)



if __name__ == "__main__":
    label_df = pd.read_csv(label_path)
    processed_df = pd.DataFrame()

    for index, row in label_df.iterrows():
        segment_id, time_to_eruption = row['segment_id'], row['time_to_eruption']
        print(f"segment id: {segment_id}")
        features = process_one_file(segment_id)
        features['time_to_eruption'] = time_to_eruption
        features = pd.DataFrame(features).T
        processed_df = pd.concat([processed_df, features], ignore_index=True)

    processed_df.to_csv('data.csv', index=False)
    
    # deal_data()

    # # for MLP
    # data = pd.read_csv('data/merged_data.csv', header=None)

    # data.columns = data.iloc[0]
    # data = data.iloc[1:]

    # scaler = StandardScaler()
    # data.iloc[:, :] = scaler.fit_transform(data.iloc[:, :])
    # data.to_csv('standard_merged_data.csv', index=False)