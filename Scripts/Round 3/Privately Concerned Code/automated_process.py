import datetime
import Utils
import pandas as pd

def adjust_NAPLAN_data():
    print("Opening file")
    data = pd.read_csv('./Datasets/Synthetic NAPLAN test/trunc.csv')

    print("Getting risks")
    risks = Utils.get_risks(data)
    starting_scores = Utils.get_risk_scores(risks)
    print("Getting heatmap")
    Utils.remap_features(data)
    starting_utility = Utils.calculate_Mat_MI(data)
    starting_scores.to_csv("Starting_scores.csv")
    Utils.plot_heatmap(starting_utility, data)

    Utils.iterative_risk_reduction(data, 8, 0)

    print("Re-calculating risk")
    risks = Utils.get_risks(data)
    final_scores = Utils.get_risk_scores(risks)
    final_utility = Utils.calculate_Mat_MI(data)

    final_scores.to_csv("Final_scores.csv")
    Utils.plot_heatmap(final_utility, data)


def test_sequence_change():
    filename = 'Datasets/Inmate Admissions/Inmate_Admissions.csv'
    col_of_interest = 'INMATEID'

    print("Opening file")
    data = pd.read_csv(filename)
    print("Getting heatmap")
    Utils.remap_features(data)
    starting_utility = Utils.calculate_Mat_MI(data)
    Utils.plot_heatmap(starting_utility, data)
    print("Creating sequence table")
    sequence_table = Utils.create_sequence_table(data, col_of_interest)
    print("Calculating risks")
    risks = Utils.get_risks(sequence_table)
    starting_scores = Utils.get_risk_scores(risks)
    current_time = datetime.datetime.now().strftime("%H%M%S")
    starting_scores.to_csv(f"{filename}_initial_sequence_risks_{current_time}.csv")
    
def test_dig_3d():
    # filename = 'Datasets/Inmate Admissions/Inmate_Admissions.csv'
    filename = 'Datasets/Inmate Admissions/small_trunc.csv'
    col_of_interest = 'INMATEID'
    max_sequence_length = 2
    print("Opening file")
    data = pd.read_csv(filename)
    print("Calculating risks")
    # header = list(data.columns)
    # header.remove(col_of_interest)
    risks = Utils.get_risks(data)
    starting_scores = Utils.get_risk_scores(risks)
    current_time = datetime.datetime.now().strftime("%H%M%S")
    deep_risks = Utils.get_DIG_3D(data, risks, col_of_interest)
    deep_scores = Utils.get_risk_scores(deep_risks)
    print("Saving files")
    starting_scores.to_csv(f"{filename}_initial_sequence_scores_{current_time}.csv")
    deep_scores.to_csv(f"{filename}_deep_scores_{current_time}.csv")
    print("Calculating utility")
    Utils.remap_features(data)
    starting_utility = Utils.calculate_Mat_MI(data)
    Utils.plot_heatmap(starting_utility, data)
    print("Reducing possible sequences")
    data = pd.read_csv(filename) 
    data = Utils.remove_extra_sequence_rows(data, col_of_interest, max_sequence_length)
    print("Re-calculating risks")
    risks = Utils.get_risks(data)
    risks.index=data.index
    starting_scores = Utils.get_risk_scores(risks)
    current_time = datetime.datetime.now().strftime("%H%M%S")
    deep_risks = Utils.get_DIG_3D(data, risks, col_of_interest)
    deep_scores = Utils.get_risk_scores(deep_risks)
    print("Saving files")
    starting_scores.to_csv(f"{filename}_end_sequence_scores_{current_time}.csv")
    deep_scores.to_csv(f"{filename}_end_deep_scores_{current_time}.csv")
    print("Recalculating utility")
    Utils.remap_features(data)
    starting_utility = Utils.calculate_Mat_MI(data)
    Utils.plot_heatmap(starting_utility, data)
    print("Done")

if __name__ == "__main__":
    # test_sequence_change()
    # (risks, deep_risks) = test_dig_3d()
    adjust_NAPLAN_data()