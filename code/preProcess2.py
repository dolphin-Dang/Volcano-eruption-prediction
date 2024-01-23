import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

data_path = 'data/train/'
label_path = 'data/train.csv'

def extract_time_domain_features(data):
    features = pd.Series()
    features['root_mean_square'] = np.sqrt(np.mean(np.square(data)))
    features['zero_crossing_rate'] = ((np.diff(np.sign(data)) != 0).sum()) / len(data)
    return features

def extract_frequency_domain_features(data):
    features = pd.Series()
    spectrum = np.fft.fft(data)
    power_spectrum = np.abs(spectrum) ** 2
    features['spectrum_energy'] = np.sum(power_spectrum)
    features['dominant_frequency'] = np.argmax(power_spectrum)
    return features


def process_one_file(seg):
    '''
    Return a features pd.Series() with 8 features of each sensor, total 80.
    '''
    df = pd.read_csv('data/train/' + str(seg) + '.csv')
    features = pd.Series()

    for i, column in enumerate(df.columns, start=1):
        if df[column].isnull().all():
            # all nan
            features['root_mean_square_'+str(i)] = np.nan
            features['zero_crossing_rate_'+str(i)] = np.nan
            features['spectrum_energy_'+str(i)] = np.nan
            features['dominant_frequency_'+str(i)] = np.nan
        else:
            # not all nan
            df[column].fillna(df[column].median(), inplace=True) # fillna anyway
            time = extract_time_domain_features(df[column])
            freq = extract_frequency_domain_features(df[column])
            features['root_mean_square_'+str(i)] = time['root_mean_square']  
            features['zero_crossing_rate_'+str(i)] = time['zero_crossing_rate'] 
            features['spectrum_energy_'+str(i)] = freq['spectrum_energy']
            features['dominant_frequency_'+str(i)] = freq['dominant_frequency']

    return features

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

    processed_df.to_csv('data/data2.csv', index=False)
    data_df = pd.read_csv('data/data.csv')
    merged_data = pd.concat([processed_df, data_df], axis=1)
    merged_data.to_csv('data/merged_data.csv', index=False)