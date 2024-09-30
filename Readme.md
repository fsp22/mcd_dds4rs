# Modelling Concept Drift in Dynamic Data Streams for Recommender Systems

This repository contains the implementation of the model presented in the paper "Modelling Concept Drift in Dynamic Data Streams for Recommender Systems".

## Repository Structure

The repository includes the following files:

- `DataStream_AMZ_exp.ipynb`: Executes experiments on the Amazon Video Games dataset.
- `DataStream_AMZ_INV_exp.ipynb`: Executes experiments on the Amazon Video Games dataset with inverted roles.
- `DataStream_MIND_exp.ipynb`: Executes experiments on the MIND dataset.
- `DataStream_YOOCHOOSE_exp.ipynb`: Executes experiments on the Yoochoose dataset.
- `datastream.py`: Contains the common code used across all experiments.
- `Readme.md`: this files.
- `Dockerfile`: docker file to build a custom image with a JupyterLab instance.
- `requirements.txt`: dependencies file.
- `MSNewsDatasetPreprocess.ipynb`: notebook for the MIND dataset preprocessing.
- `PreprocessYoochooseDataset.ipynb`: notebook for the Yoochoose dataset preprocessing.



## Getting Started

To run the experiments, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/fsp22/mcd_dds4rs.git
   cd mcd_dds4rs
   # fix permission for docker environment
   chmod 777 your-repo-name
   chmod 666 your-repo-name/*
   ```

2. **Build the docker image (optional)**

    ```
    docker build --rm --tag my-ds-image .
    ```

    The docker contains required libraries for this project. The dependecies are defined in the requirements.txt.

3. **Executes the Jupyter notebooks**.

    ```bash
    docker run -it --rm -p 18888:8888 -v "${PWD}":/home/jovyan/work my-ds-image start-notebook.py --ip 0.0.0.0 --IdentityProvider.token=''
    ```

4. **Open the browser at**

    ```
    http://localhost:18888/lab
    ```

Now you can execute the notebooks.



### Data preprocessing

For the dataset Yoochoose and MIND, the source dataset files must be pre-processed with the notebooks:

- MSNewsDatasetPreprocess.ipynb

- PreprocessYoochoseDataset.ipynb

