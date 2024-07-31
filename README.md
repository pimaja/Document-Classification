1. OVERVIEW
This is a Document Classification project in Python, based on the Kaggle challenge https://www.kaggle.com/datasets/sachinsharma1123/document-classification/data, and further developed for inferencing in production.

It consists of 2 main files:
- **development.ipynb** Jupyter Notebook where the data is analyzed and different machine learning models tested. Results show that BERT-Mini gives the best performance without the need to over- and undersample the imbalanced dataset, which is required for the classical machine learning models. All structures of BERT-Mini and its fine-tuning can be found in this notebook.
- **main.py** script for production which takes a text file as input and outputs the predicted category for the model. For prediction it uses a checkpoint of a fine-tuned BERT-Mini in development.ipynb.

Other files:
- **checkpoint** - checkpoint of a fine-tuned BERT-Mini for inferencing only (optimizer.pt and scheduler.pt were removed to decrease the size of the checkpoint). Keep this in the same directory as the **main.py** to be able to run the script.
- **test_inputs** - some test inputs used to test **main.py**.
- **requirements.txt** - required Python packages for running **main.py** script.
- **file.txt** - contains the data used for fine-tuning in **development.ipynb**. Downloaded from Kaggle https://www.kaggle.com/datasets/sachinsharma1123/document-classification/data. Keep this file in the same directory as development.ipynb to be able to run the notebook.

2. INSTALLATIONS
For **main.py** (recommended):
- create a Conda environment
  ```
  conda create -n doc-classification python=3.10 -y
  ```
- activate this Conda environment
  ```
  conda activate doc-classification
  ```
- install requirements.txt with pip
  ```
  pip install -r requirements.txt
  ```

For **development.ipynb**:
- the notebook can be ran on Google Collab, Kaggle or own instance with Python and Jupyter Notebook installed.
- pip installation command for all needed packages is included in the development.ipynb.
