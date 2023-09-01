import pandas as pd
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from aif360.algorithms.preprocessing import Reweighing


def fairness_dataset():
    col_names = [
        'Status of exisiting checking account',
        'Duration in month',
        'Credit history',
        'Purpose',
        'Credit amount',
        'Savings account/bonds',
        'Present employment since',
        'Installment rate in percentage of disposable income',
        'sex',
        'Other debtors / guarantors',
        'Present residence since',
        'Property',
        'Age in years',
        'Other installment plans',
        'Housing',
        'Number of existing credits at this bank',
        'Job',
        'Number of people being liable to provide maintenance for',
        'Telephone',
        'foreign worker',
        'Target'
    ]
    
    categorical_features = [
        'Status of exisiting checking account',
        'Credit history',
        'Purpose',
        'Savings account/bonds',
        'Present employment since',
        'sex',
        'Other debtors / guarantors',
        'Property',
        'Other installment plans',
        'Housing',
        'Job',
        'Telephone',
        'foreign worker'
    ]

    numerical_features = [
        'Duration in month',
        'Credit amount',
        'Installment rate in percentage of disposable income',
        'Present residence since',
        'Age in years',
        'Number of existing credits at this bank',
        'Number of people being liable to provide maintenance for',
        'Target'
    ]

    protected_attribute_names = ['sex']

    df = pd.read_csv('./Dataset/German-Dataset.csv', names=col_names, index_col=False, header=None)


    one_hot = pd.get_dummies(df[categorical_features], dtype=int)
    df = df.drop(categorical_features, axis=1)
    df = df.join(one_hot)
    print(df.head)

    protected_attribute_names = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94'
    ]

    dataset = StandardDataset(
        df=df,
        label_name='Target',
        favorable_classes=[1],
        protected_attribute_names=protected_attribute_names,
        privileged_classes=[lambda x: x == 1]
    )
    
    privileged_groups = [{'sex_A91': 1}]
    unprivileged_groups = [{'sex_A91': 0}]

    dataset_orig_train, dataset_orig_test = dataset.split([0.7], shuffle=True)

    metric_original_train = BinaryLabelDatasetMetric(dataset=dataset_orig_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print(f'Metrica: {metric_original_train.mean_difference()}')

    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transf_train = rw.fit_transform(dataset_orig_train)

    metric_transf_train = BinaryLabelDatasetMetric(dataset=dataset_transf_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    print(f'Metrica: {metric_transf_train.mean_difference()}')
    
fairness_dataset()