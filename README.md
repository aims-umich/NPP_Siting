# Brownfield-CPP_to_NPP_Transition
The dataset and codes which has been used while analyzing the suitability of NPP siting in U.S. brownfields and CPP sites. 

## Paper

Erdem, O., Daley, K., Hoelzle, G., & Radaideh, M. I. (2025). Multi-objective combinatorial methodology for nuclear reactor site assessment: A case study for the United States. Energy Conversion and Management: X, 100923. https://www.sciencedirect.com/science/article/pii/S2590174525000558

## 🛠️ Environment Installation

The environment.yaml files are presented for different directories. All directories use the base environment file, with the exception of model hyperparameter tuning, and model generation directories. Separate environment files are provided for these 2 steps.

To set up environments for this project, follow these steps:

```bash
# 1. Create a new conda environment for data processing in NPP_Siting directory
conda env create -f NPP_env_environment.yaml

# 2. Create a new conda environment for model hyperparameter tuning in NPP_Siting/Model Training/Model_grid_tuning/ directory
conda env create -f Model_Training/Grid_tuning/tfcpu_environment.yml

# 3. Create a new conda environment for model training tuning in NPP_Siting/Model Training/ directory
conda env create -f Model_Training/tfgpu_environment.yml

```

## How to generate the results

The files have been separated to 3 parts. Preprocessing (1), processing (2), postprocessing (3) and neural network training (4) steps are outlined below. For each folder, running the accompanying Python or Jupyter Notebook files creates the datasets, processes and postprocesses the results.

- Step 1: Run the data generation of the brownfields in the "NPP_Siting/Preprocessing/Brownfields" directory. This script merges the data coming from different computations and GeoDataBase layers. 

```bash
nohup python merge_data.py &
```

- Step 2: Run the data generation of the coal power plants in the "NPP_Siting/Preprocessing/CPPs" directory. This script merges the data coming from different computations and GeoDataBase layers. 

```bash
nohup jupyter nbconvert --to notebook --execute --inplace cpp_preprocessor.ipynb > output.log 2>&1 &
```

- Step 3: Run the processor script in "NPP_Siting/Processing" directory. This script includes the main data processing code, it requires very high amount of computational power.

```bash
nohup python Brownfield_NS_v13.py &
```

- Step 4: Run the postprocessor script in "NPP_Siting/Postprocessing" directory. The codes in this Jupyter Notebook generates all the siting metric data, site objective contributions, and the figures shown in the paper.

```bash
nohup jupyter nbconvert --to notebook --execute --inplace postprocessor.ipynb > output.log 2>&1 &
```

The data of postprocessing is exported at every step and can be found in the "NPP_Siting/Postprocessing" directory after running this Jupyter Notebook.

- Step 5: Run the model first layer tuner in the "NPP_Siting/Model_Training/Grid_tuning/ConcNN_First_Layer_Tuning" directory.

```bash
nohup python Model_Grid_Tuner.py &
```

- Step 6: Run the model second layer tuner in the "NPP_Siting/Model_Training/Grid_tuning/ConcNN_and_LUT-NN_Second_Layer_Hypertune" directory.

```bash
nohup python Model_Grid_Tuner.py &
```

- Step 7: Train the models in the "NPP_Siting/Model_Training" directory.

```bash
nohup jupyter nbconvert --to notebook --execute --inplace Interpolator.ipynb > output.log 2>&1 &
nohup jupyter nbconvert --to notebook --execute --inplace ConcNN.ipynb > output.log 2>&1 &
nohup jupyter nbconvert --to notebook --execute --inplace LUT-NN.ipynb > output.log 2>&1 &
```
