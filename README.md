# GraphDTI: Graph Neural Networks for Drug-Target Interaction Prediction

## 🔬 Overview
GraphDTI is a machine learning pipeline for predicting Drug-Target Interactions (DTI) using Graph Neural Networks (GNNs). It compares GNNs with traditional models like XGBoost and provides a reproducible framework for computational drug discovery.

## 📂 Repository Structure
```
📁 GraphDTI  
│── 📂 data/              # Dataset (processed or raw)  
│── 📂 src/               # Source code (model training, evaluation)  
│── 📂 notebooks/         # Jupyter notebooks for analysis  
│── 📂 results/           # Trained models, performance reports  
│── requirements.txt      # Dependencies  
│── environment.yml       # Conda environment setup  
│── README.md            # Project documentation  
│── LICENSE              # Open-source license  
```

## 🚀 Installation

### **Using pip**
```bash
git clone https://github.com/rajendraprasadchepuri/GraphDTI.git
cd GraphDTI
pip install -r requirements.txt
```

### **Using Conda**
```bash
conda env create -f environment.yml
conda activate graphdti
```

## 📊 Training the Model
Run the following script to train the model:
```bash
python src/train.py --model GNN --epochs 50
```

## 📈 Results
| Model | Accuracy | AUC | F1-Score |
|--------|------------|------|----------|
| XGBoost | 87.5% | 0.89 | 0.85 |
| GNN | **91.2%** | **0.94** | **0.90** |

## 📜 License
This project is licensed under the MIT License.
