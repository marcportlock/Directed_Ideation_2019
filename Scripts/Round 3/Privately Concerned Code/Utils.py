import datetime
import klcalculator
import itertools
import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
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


def calc_MI(X,Y,bins):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]
    
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    
    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def remap_features(data):
    header = data.columns
    for col in header:
        print(f"Processing {col}")
        labels, _ = data[col].factorize()
        data[col] = labels

def plot_heatmap(matMI, data):
    plt.clf()
    sns_plot=sns.heatmap(matMI,
            cmap="YlGnBu_r",
            vmin=0,
            vmax=1,
            annot=True, linewidths=.5, fmt='.2f'
            )
    sns_plot.set_xticklabels(data.columns, rotation=45, ha="right")
    sns_plot.set_yticklabels(data.columns, rotation=0)
    sns_plot.set_title("Mutual Information (MI)")
    plt.tight_layout()
    fig = sns_plot.get_figure()
    fig.savefig(f'heatmap{datetime.datetime.now().strftime("%H%M%S")}.png')    

def get_risk_scores(data):
    combs = data.columns
    combined_risks = []
    for comb in combs:
        minimum = data[comb].min()
        maximum = data[comb].max()
        q1 = data[comb].quantile(0.25)
        q3 = data[comb].quantile(0.75)
        avg = data[comb].mean()
        med = data[comb].median()
        tup = [minimum, q1, avg, med, q3, maximum]
        combined_risks.append(tup)
    
    df = pd.DataFrame(combined_risks)
    df.index = combs
    df.columns = ['Min', 'Q1', 'Avg', 'Med', 'Q3', 'Max']

    return df 

    
def calculate_Mat_MI(data):
    A = data.values
    bins = 2
    n = A.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        print(f"Processing column {ix}")
        for jx in np.arange(n):
            matMI[ix,jx] = calc_MI(A[:,ix], A[:,jx], bins)

    return matMI

def normal_process(data):
    print("Getting risks")
    risks = get_risks(data)
    scores = get_risk_scores(risks)
    scores.to_csv(f'heatmap{datetime.datetime.now().strftime("%H%M%S")}.csv')
    print("Getting heatmap")
    remap_features(data)
    matMI = calculate_Mat_MI(data)
    plot_heatmap(matMI, data)

# def reduce_risky_feature(data, column, reduction_amount):
#     # some sort of even aggregation. Assumes remapped data
#     initial = len(data[column].unique())
#     new_groups = math.ceil(initial / reduction_amount)
#     for i in new_groups:
#         pass

# assumes remapped data
def iterative_risk_reduction(data, allowable_risk, max_iterations):
    header = data.columns
    no_change = [False] * len(header)
    iteration_number = 1
    while not all(no_change):
        print(f"Iterative risk re-calc {iteration_number}")
        risks = get_risks(data)
        scores = get_risk_scores(risks)
        for idx, col in enumerate(header):
            print(f"Updating column {col}")
            if no_change[idx] == False:
                col_score = scores.at[col, 'Max']
                if col_score < allowable_risk:
                    no_change[idx] = True
                    print("Column finished")
                else:
                    vals = data[col].value_counts()
                    vals.sort_values(inplace=True)
                    change_list = list(vals.index[vals==vals[0]])
                    change_list = change_list + list(vals.index[vals==vals[1]])

                    new_val = data[col].max() + 1
                    data[col] = data[col].replace(change_list, new_val)

        iteration_number = iteration_number + 1

        if iteration_number >= max_iterations:
            break
            

# def reduce_risky_individuals(data, column):
#     # some sort of uneven aggregation. Assumes remapped data
#     pass

# def reduce_all_risks(data, scores):
#     # check type of risk for each column and iterate until risk is acceptable
#     header = data.columns
#     change = [True] * len(header)
#     while all(change):
#         for idx, col in enumerate(header):
#             if change[idx] == True:
#                 col_score = scores.loc(col)
#                 if  col_score['Max'] > 9 and col_score['Min'] > col_score['Max'] - 3:
#                     reduce_risky_feature(data, col, (10^(col_score['Max'] - 8)) / 2)
#                 elif col_score['Max'] > 8 and col_score['Min'] < col_score['Max'] - 3: 
#                     reduce_risky_individuals(data, col)
#                 else:
#                     change[idx] = False


def create_sequence_table(data, col_of_interest):
    print(datetime.datetime.now())

    number_of_cases = len(data[col_of_interest].unique())
    index = data[col_of_interest].unique()
    header = list(data.columns)
    header.remove(col_of_interest) 
    header = [col_of_interest] + header
    data = data[header].values.tolist()
    data.sort()
    groups = itertools.groupby(data, lambda x:x[0])
    new_table = pd.DataFrame()
    
    current_name = 1
    max_group_len = 0
    for name, group_gen in groups:
        group = list(group_gen)
        if len(group) > max_group_len:
            max_group_len = len(group)
        if len(group) > 41:
            group = group[0:41]
        
        if current_name % 10 == 0 or current_name == 1: 
            print(f"{current_name} of {number_of_cases}")

        for n in range(1, len(header)):
            col = [x[n] for x in group]
            for i in [40]: # range(len(group[col].unique())):
                if len(group) < i:
                    continue

                sequences = itertools.combinations(col, i)
                for seq in sequences:
                    sequence = f"{header[n]}_{'_'.join(str(x) for x in seq)}"
                    if sequence not in new_table.columns:
                        new_table[sequence] = pd.Series([0] * number_of_cases, index=index)

                    new_table.at[name, sequence] = 1
        
        current_name = current_name + 1

    print(datetime.datetime.now())
    print(max_group_len)

    return new_table

def get_DIG_3D(data, original_risks, col_of_interest):
    print("Getting DIG 3D")
    print(datetime.datetime.now())

    all_indices = list(data.index)
    all_group_labels = data[col_of_interest].unique()
    number_of_groups = len(all_group_labels)
    header = list(data.columns)
    header.remove(col_of_interest) 
    header = [col_of_interest] + header
    data_list = data[header].reset_index().values.tolist()

    data_list.sort(key=lambda x:x[1])
    groups = itertools.groupby(data_list, lambda x:x[1])

    list_of_frames = []
    print("Processing groups")
    groups_processed = 1
    for _, group_gen in groups:
        if groups_processed % 100 == 0 or groups_processed == 1: 
            print(f"{groups_processed} of {number_of_groups}")
        group = list(group_gen)
        max_sequence_length = len(list_of_frames)
        current_sequence_length = len(group)
        if current_sequence_length > max_sequence_length:
            for _ in range(current_sequence_length - max_sequence_length):
                blank_array = np.full((len(all_indices), len(header)), "NON-UNIQUE-VALUE")
                frame = pd.DataFrame(blank_array, columns=header, index=all_indices)
                list_of_frames.append(frame)

        for n in range(len(group)):
            frame = list_of_frames[n]
            current_row = group[n]
            row_index = current_row[0]
            frame.at[row_index, col_of_interest] = current_row[1] 
            for feature in range(2, len(header) + 1):
                current_group = [str(x[feature]) for x in group[0:n]]
                new_value = '_'.join(current_group)
                if new_value != '': 
                    frame.at[row_index, header[feature - 1]] = new_value

        groups_processed = groups_processed + 1

    print("Copying original risks")
    deep_risks = original_risks.copy()
    print("Updating risks with DIG")
    for n, df in enumerate(list_of_frames):
        print(f"{n} of {len(list_of_frames)}")
        print("Calculating risks")
        new_risks = get_risks(df)
        new_risks.index = all_indices
        print("Updating risk values")
        for head in header:
            for idx in all_indices:
                z = deep_risks.at[idx, head] 
                s = new_risks.at[idx, head] 
                deep_risks.at[idx, head] = max([s, z])
    return deep_risks

def remove_extra_sequence_rows(data, col_of_interest, max_sequence_length):
    number_of_groups = len(data[col_of_interest].unique())
    header = list(data.columns)
    header.remove(col_of_interest) 
    header = [col_of_interest] + header
    data_list = data[header].reset_index().values.tolist()

    data_list.sort(key=lambda x:x[1])
    groups = itertools.groupby(data_list, lambda x:x[1])

    print("Processing groups")
    groups_processed = 1
    for _, group_gen in groups:
        if groups_processed % 100 == 0 or groups_processed == 1: 
            print(f"{groups_processed} of {number_of_groups}")
        group = list(group_gen)
        if len(group) > max_sequence_length:
            indices = [x[0] for x in group[max_sequence_length - 1:-1]]
            for i in indices:
                data = data.drop(i, axis=0)

        groups_processed = groups_processed + 1

    return data