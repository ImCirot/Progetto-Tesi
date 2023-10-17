import ucimlrepo
import pandas as pd


def load_dataset():
    dataset = ucimlrepo.fetch_ucirepo(id=222)

    df = dataset.data.original
    drop_features = ['poutcome','contact','pdays']

    df.drop(drop_features,axis=1,inplace=True)

    feature_names = df.columns.tolist()

    feature_names.remove('y')
    df.to_csv('./Bank Marketing Dataset/original_dataset.csv', index_label='ID')

    df_dummies = pd.get_dummies(df[feature_names],dtype=int)
    df = df.drop(feature_names, axis=1)
    df = df.join(df_dummies)

    df['y'] = df['y'].replace('no',0)
    df['y'] = df['y'].replace('yes',1)

    df.to_csv('./Bank Marketing Dataset/dataset.csv')

load_dataset()