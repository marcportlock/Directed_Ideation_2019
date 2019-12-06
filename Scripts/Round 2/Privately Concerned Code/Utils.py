import datetime
import klcalculator
import itertools
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
        if col == "ID":
            continue

        print(f"Processing {col}")
        match_dic = data[col].unique()
        length = len(match_dic)
        for i in range(length):
            data[col].replace(match_dic[i], i+1, inplace=True)

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