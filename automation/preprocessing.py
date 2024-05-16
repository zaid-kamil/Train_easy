import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(data):
    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Encoding categorical variables
    data_encoded = pd.get_dummies(data_filled)
    
    # Scaling numerical features
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_encoded), columns=data_encoded.columns)
    
    return data_scaled