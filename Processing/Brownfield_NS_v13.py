from NS_funcs import myFirstPareto
from joblib import Parallel, delayed
from itertools import combinations, islice
from sklearn.preprocessing import MinMaxScaler
from functools import partial
from time import time
from math import comb
import numpy as np
import pandas as pd

csv_path = "Brownfield_Data.csv"
df_in_out = pd.read_csv(csv_path)
#data_array = np.loadtxt("bf_cpp_preprocessed.txt")

df_filtered = df_in_out[~df_in_out.duplicated(subset=["x", "y"])]
df_filtered = df_filtered.reset_index(drop=True)

df_nometric = df_filtered.iloc[:, :-1]

# The functions described here uses the threshold values and linear scaling for the mileage described in the proximity section of the STAND tool.
def process_pop_column(df, col):
    df.iloc[:, col] = np.where(df.iloc[:, col] < 4, 0, df.iloc[:, col])
    max_val = df.iloc[:, col].max()
    if max_val > 4:
        df.iloc[:, col] = np.where(df.iloc[:, col] > 4, (df.iloc[:, col] - 4) / (max_val - 4), 0)
    return df

def process_ret_colum(df, col):
    max_val = df.iloc[:, col].max()
    if max_val > 1:
        max_val = 1
    df.iloc[:, col] = np.where(df.iloc[:, col] < 1, (max_val - df.iloc[:, col]) / max_val, 0)
    return df
    
def process_subs_colum(df, col):
    max_val = df.iloc[:, col].max()
    df.iloc[:, col] = (max_val - df.iloc[:, col]) / max_val
    return df

def process_road_colum(df, col):
    df.iloc[:, col] = np.where(df.iloc[:, col] < 1, 1, df.iloc[:, col])
    max_val = df.iloc[:, col].max()
    if max_val > 1:
        df.iloc[:, col] = np.where(
            df.iloc[:, col] > 1,
            (max_val - df.iloc[:, col]) / (max_val - 1),
            df.iloc[:, col])
        return df

# The coordinate columns are rounded to 2 decimal places and duplicate coordinates are removed.
def round_and_remove_duplicates(df, decimals=2):
    # df_rounded = df.copy()
    df.iloc[:, 1:3] = df.iloc[:, 1:3].round(decimals)
    df_unique = df.drop_duplicates(subset=df.columns[1:3])
    return df_unique

def negate_columns(df, negative_cols):
    df.iloc[:, negative_cols] = -df.iloc[:, negative_cols]
    return df

def minmax_scale_columns(df, start):
    scaler = MinMaxScaler()
    df.iloc[:, start:] = scaler.fit_transform(df.iloc[:, start:])
    return df

cpp_path = "CPP_Data_Complete.csv"
cpp_df = pd.read_csv(cpp_path)

df_bff_cp = pd.concat([df_nometric, cpp_df], ignore_index=True)

df_bff_cp.to_csv('BF_CPP_DATA.csv', index=False)
df_bff_cp.iloc[:,0].to_csv('registry_ids.csv', index=False)

data = df_bff_cp
# The negative objectives in the BF_CPP_DATA.csv file have been negated in the next lines. 
# These are state nuclear restrictions, pop_weight_svi, state_const_labor_wage, protected_lands, count_hazard_facs 
negative_columns = [5, 9, 11, 12, 13]
data = process_pop_column(data, 21)
data = process_ret_colum(data, 22)
data = process_subs_colum(data, 24)
data = process_road_colum(data, 25)
data = negate_columns(data, negative_columns)

data = minmax_scale_columns(data, start=5) 

# Only the objectives are kept for the next step
data_array = data.iloc[:, 5:].values

num_locs = len(data_array)
num_obj = len(data_array[0])

# myFirstPareto function takes an array of objectives, takes a combination for this array, and calculates the first (best) pareto front for this combination of objectives.
# Partial function has been used for creating the function which will be used in parallel computation. 
comb_to_pareto = partial(myFirstPareto, data_array)

ranks=[i for i in range(num_locs)]
out_data_indices=[i for i in range(0,num_obj)]

# The files to count the non-normalized site observation rates and objective contributions 
w_file_name="recorder.txt"
obj_contr_file="obj_contributions.txt"

# Cores for parallel combination, and batch size. The batch size is required since calculation of a combination size in a single batch is not possible. 
# There are over 700k combinations for C(22,11). The program holds a 16k x 22 matrix for 700k different calculations, resulting in memory overflow. 
# This computation is separated to batches. 
ncores = 360
batch_size = ncores
for i in range(0, 22):  # Iterating over combination sizes
    t_init = time()
    objective_contributions = np.zeros((num_locs, num_obj), dtype=int)
    comb_M = i + 1
    distribution = np.zeros(num_locs, dtype=int)

    # Calculate total combinations and total batches
    total_combinations = comb(num_obj, comb_M)
    total_batches = (total_combinations + batch_size - 1) // batch_size  # Ceiling division

    # Generator for combinations
    combs = combinations(out_data_indices, comb_M)

    # Process in batches
    batch_count = 0
    while True:
        # Take the next batch of combinations
        comb_batch = list(islice(combs, batch_size))
        if not comb_batch:
            break  # No more combinations

        # Run Pareto calculations in parallel for this batch
        with Parallel(n_jobs=ncores) as parallel:
            pareto_fronts = parallel(delayed(comb_to_pareto)(comb) for comb in comb_batch)

        # Update distribution and objective_contributions for this batch
        for pareto_front in pareto_fronts:
            for front_elements in range(len(pareto_front)):
                rank_num = pareto_front[front_elements][0]
                distribution[ranks[rank_num]] += 1

            for front in pareto_front:
                location = front[0]
                rec_comb = front[1]
                for rec_objective in rec_comb:
                    objective_contributions[location][rec_objective] += 1

        # Increment the batch count and print progress
        batch_count += 1
        progress_percent = (batch_count / total_batches) * 100
        print(f"Combination size {comb_M}: {progress_percent:.2f}% complete.")

    # Write objective contributions to file
    with open(obj_contr_file, "a") as file:
        file.write(f"The {i+1} combinations contributions:\n")
        for row in objective_contributions:
            file.write(" ".join(map(str, row)) + "\n")

    # Write distribution to file
    line = f"Elapsed {((time() - t_init) / 60):.2f} mins. The {i+1} combinations result:\n"
    line += " ".join(map(str, distribution)) + "\n"
    with open(w_file_name, "a") as opened_file:
        opened_file.write(line)

    print(f"Combinations of {i+1} are complete. Elapsed minutes: {(time() - t_init) / 60:.2f}")

print("Processing complete.")