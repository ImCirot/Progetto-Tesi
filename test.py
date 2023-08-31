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

    protected_attribute_names = ['sex']

    df = pd.read_csv('./Dataset/German-Dataset.csv', names=col_names, index_col=False, header=None)

    dataset = StandardDataset(
        df=df,
        label_name='Target',
        favorable_classes=[1],
        categorical_features=categorical_features,
        protected_attribute_names=['Age in years'],
        privileged_classes=[lambda x: x<=25],

    )

    print(dataset)

    dataset_orig_train, dataset_orig_test = dataset.split([0.7], shuffle=True)

    privileged_groups = [{'Age in years': 0}]
    unprivileged_groups = [{'Age in years': 1}]

    metric_original_train = BinaryLabelDatasetMetric(dataset=dataset_orig_train, unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
    
    print(f'- {metric_original_train.mean_difference()}')

    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)

    metric_transf_train = BinaryLabelDatasetMetric(dataset=dataset_transf_train, unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    print(f'- {metric_transf_train.mean_difference()}')



fairness_dataset()