import pand as as pd  
import numpy as np  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split  
 
 
def load_dataset(path):  
 
    df = pd.read_csv(path)  
 
    df = df.dropna()  
 
    return df  
 
 
 
def preprocess(df, label_column='Label'):  
 

    X = df.drop(columns=[label_column]).values  
    y = df[label_column].values  
 
    scaler = MinMaxScaler()  
 
    X = scaler.fit_transform(X)  
 
    return X, y  
 
 
 
def create_sequences(X, y, seq_len=20):  
 
    X_seq = [] 
    y_seq = []  
 
    for i in range(len(X) - seq_len):  
 
        X_seq.append(X[i:i+seq_len])  
        y_seq.append(y[i+seq_len])  
 
    return np.array(X_seq), np.array(y_seq)
