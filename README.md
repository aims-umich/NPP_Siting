# Brownfield-CPP_to_NPP_Transition

https://www.sciencedirect.com/science/article/pii/S2590174525000558

The dataset and codes which has been used while analyzing the suitability of NPP siting in U.S. brownfields and CPP sites. 

The preprocessing (1), processing (2), postprocessing (3) and neural network training (4) steps are outlined below:

1) Data generation of the brownfields and CPPs are done separately in this folder. 

1.1) The preprocessing starts at Data Generation. The raw data from GeoDataBase files exist in bf folder. "merge_data.py" merges these data from different data acquisition steps. 

1.2) In cpp folder, the raw data of coal power plants are merged. Then the codes "coord_to_state_county.py", "state_county_to_fips.py", and "merge_objectives_and_fips.py" merge the FIPS codes of each CPP with the raw datasets.

2) In Data Processing folder, Brownfield_Data.csv and CPP_Data_Complete.csv files hold the merged files. "Brownfield_NS_v13.py" code starts by merging them, then it negates the negative objectives, processes the proximity objectives to match them with the objectives presented in the STAND tool, min-max scales the objectives. Then it runs this preprocessed dataset through the methodology described in the paper "Multi-objective Combinatorial Methodology for Nuclear Reactor Site Assessment: A Case Study for the United States".

3) The Data Visualization takes recorder.txt and obj_contributions.txt files. It uses "recorder_reader.ipynb" file to read the recorder file. This file normalizes the file for each combination length, creates the siting metric, and creates the 3D plot in the paper. Then after the best site is found, "2d_site_plotter.ipynb" creates the 2D plot for the best site result. "process_obj_contributions.py" creates the summed normalized objective contributions. "show_bar_chart_for_importances.py" plots the objective contributions of best 6 sites. "create_filled_xy_data.ipynb" reverses the coordinate rounding. "create_xy_data_with_results.py" creates the new, complete dataset with objectives and processing results. "get_final_res_for_mapping.py" gets the siting metric and coordinates for the top results. "mapper.py" creates the map for the best sites. "plot_computation_time.ipynb" prints and plots the computation times. "get_variance_change.py" plots the variance change w.r.t. combination length.

4) Model Training folder uses "Complete_BF_CPP_XY_Data.csv" file. This folder includes the ConcNN and LUT-NN scripts. The grid tuning directory and model training directory has separate virtual environments.

4.1) Model_grid_tuning folder has the grid hyperparameter tuning script "Model_Grid_Tuner.py" for two cases, first layer of ConcNN, and LUT-NN and second layer of ConcNN. 

4.2) "Interpolator_only.ipynb" file shows the structure of the interpolator used in LUT-NN model. 

4.3) "ConcNN_Training_2000_ep_256_bs" file shows the codes used to generate the ConcNN model described in the paper. Its model is given as "Model_ConcNN_First_Part_2000_256.keras". The first part of the model has been trained in "ConcNN_First_Part_Metrics.ipynb" for comparison with the interpolation, and its model is given in "Model_ConcNN_First_Part_2000_256.keras".

4.4) "LUT-NN_training_1000_ep_16_bs" file shows the codes used to generate the LUT-NN model described in the paper. Its model is given as "Model_LUT-NN_1000_16.keras".
