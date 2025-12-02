# House Price Prediction

A complete machine-learning pipeline for predicting house prices using scikit-learn.

## Setup

```bash
pip install -r requirements.txt
python train.py --data-path data/train.csv --model-out models/model.joblib
python predict.py --model-path models/model.joblib --input-file data/test.csv --output-file predictions.csv
