import pandas as pd
import seaborn as sns

col_names = [
    'Status of exisiting checking account',
     'Duration in month',
     'Credit history',
     'Purpose',
     'Credit amount',
     'Savings account/bonds',
     'Present employment since',
     'Installment rate in percentage of disposable income',
     'Personal status and sex',
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
     'foreign worker'
]
dataset = pd.read_csv('./Dataset/German-Dataset.csv', index_col=False, header=None, names=col_names)

print(dataset)