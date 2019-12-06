import klcalculator
import pandas as pd
from statistics import mean
from Utils import get_combined_feature_risks

data = pd.read_csv('./Datasets/Synthetic NAPLAN test/NAPLAN_synthetic.csv')
for col in ['Surname', 'First_Name']:
    data = data.drop(col, axis=1)

data['DOB'] = pd.to_datetime(data['DOB'])
data['DOB'] = data['DOB'].dt.strftime('%m/%Y')
risks = get_combined_feature_risks(data)
