import numpy as np
import pandas as pd


def formatting_data():
    
    df = pd.read_csv('./Adult Dataset/adult_dataset.CSV')

    print(df.head)

formatting_data()
