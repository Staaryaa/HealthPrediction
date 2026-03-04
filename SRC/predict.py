'''

FEATURES : 

AGE - CONTINOUS
BMI - CONTINOUS                
SYSTOLIC - CONTINOUS            USE SCALER AND TRANSFORM AND ADD TO INPUT
DIASTOLIC - CONTINOUS
STRESS_LEVEL - CONTINOUS

GENDER - CATEGORICAL - MALE , FEMALE , OTHER
SMOKING - CATEGORICAL - never , former , current  - CREATE SEPERATE COLS AND ADD TO INPUT
DRINKING - CATEGORICAL - never , occasional , regular, heavy
EXERSIZE - CATEGORICAL - sedentary , light , moderate , intense

FAMILY HISTORY DIABTES - BINARY
FAMILY HISTORY HEART - BINARY  - AS IS TO INPUT
FAMILY HISTORY OBESITY - BINARY


from the user we get age in integer , bmi , systolic and diastolic bp and stress level
we ask for gender :  male female or other
we ask smoking : never , former , current
we ask drinking : never , occasional , regular, heavy
we ask exersize : sedentary , light , moderate , intense
#my_model = keras.models.load_model(FINAL_MODEL_PATH,compile=False)
we ask history of diabetes , history of heart, we ask history of 

'''

import numpy as np
import pandas as pd
import keras
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler


SRC_PATH = Path(__file__).resolve().parent 
ROOT_PATH = SRC_PATH.parent
PRO_DATA_PATH = ROOT_PATH / "DATA" / "processed"
MODEL_PATH = ROOT_PATH/ "MODEL"
FINAL_MODEL_PATH = MODEL_PATH/"health_model.keras"

scalar = StandardScaler()



with open(PRO_DATA_PATH/"preprocessor.pkl" , "rb") as f:
    b = pickle.load(f)
    scalar = b["scaler"] 
    


def transform_input(df):

    BINARY = ['family_history_diabetes', 'family_history_heart', 'family_history_obesity']

    CONTINUOUS = [
        'age',
        'bmi',
        'systolic_bp',
        'diastolic_bp',
        'stress_level'
    ]

    CATEGORICAL = {
        'gender':   ['Male', 'Female', 'Other'],
        'smoking':  ['Never', 'Former', 'Current'],
        'drinking': ['Never', 'Occasional', 'Regular', 'Heavy'],
        'exercise': ['Sedentary', 'Light', 'Moderate', 'Intense'],
    }

    cont_vals = pd.DataFrame(
        scalar.transform(df[CONTINUOUS]),
        columns=CONTINUOUS, index=df.index
    )

    cat_dfs = []
    for col, categories in CATEGORICAL.items():
        for cat in categories:
            cat_dfs.append(pd.Series(
                (df[col] == cat).astype(int),
                name=f"{col}_{cat}", index=df.index
            ))
    cat_vals = pd.concat(cat_dfs, axis=1)

    bin_vals = df[BINARY].reset_index(drop=True)
    cont_vals = cont_vals.reset_index(drop=True)
    cat_vals = cat_vals.reset_index(drop=True)

    transformed_input = pd.concat([cont_vals, cat_vals, bin_vals] , axis=1)

    return transformed_input

'''

def make_predictions(input_from_user):

    predictions = my_model.predict(input_from_user)

    level = ["LOW" , "MEDIUM" , "HIGH"] 

    diabetes_risk = np.argmax(predictions[0])
    heartdisease_risk = np.argmax(predictions[1])
    obesity_risk = np.argmax(predictions[2])

    print(f"RISK FOR DIABETES : {level[diabetes_risk]}")
    print(f"RISK FOR HEART DISEASE : {level[heartdisease_risk]}")
    print(f"RISK FOR OBESITY : {level[obesity_risk]}")

'''



















