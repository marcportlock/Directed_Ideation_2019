import klcalculator
import itertools
import pandas as pd
from statistics import mean, median

# takes a pandas dataframe
def get_combined_feature_risks(data):
        features = list(data.columns.values)
        combined_risks = []
        combined_one_cols = []
        for i in range(2, 3): # change this as required
                combs = list(itertools.combinations(features, i))
                for comb in combs:
                        cols = data[list(comb)].values.tolist()
                        one_col = []
                        for row in cols:
                                line = ''
                                for item in row:
                                        line = line + str(item)
                                one_col.append(line)
                        combined_one_cols.append(one_col)
                frame = pd.DataFrame(combined_one_cols, index=combs)
                frame = frame.transpose()
                risks = get_risks(frame)
                for comb in combs:
                        minimum = risks[comb].min()
                        maximum = risks[comb].max()
                        q1 = risks[comb].quantile(0.25)
                        q3 = risks[comb].quantile(0.75)
                        avg = risks[comb].mean()
                        med = risks[comb].median()
                        tup = [minimum, q1, avg, med, q3, maximum]
                        combined_risks.append(tup)

        risk_frame = pd.DataFrame(combined_risks, index = combs, columns = ['Min', 'Q1', 'Mean', 'Med', 'Q3', 'Max']) # change this, this will break with more than one combination length

        return risk_frame

# takes a pandas dataframe
def get_risks(data):
        features = list(data.columns.values)
        dataset = list(zip(*(data[fn].tolist() for fn in features)))
        risks = klcalculator.find_risks_for_records(dataset)
        risks_frame = pd.DataFrame(risks, columns=features)
        
        return risks_frame