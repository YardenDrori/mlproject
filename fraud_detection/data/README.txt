The dataset file (creditcard.csv) is not included in this repository because it exceeds 100MB.

HOW TO ADD THE DATA
-------------------

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Download creditcard.csv (either manually or via the Kaggle CLI):

   pip install kaggle
   # Place your Kaggle API token at ~/.kaggle/kaggle.json
   kaggle datasets download -d mlg-ulb/creditcardfraud
   unzip creditcardfraud.zip -d fraud_detection/data/

3. Confirm the file is in the right place:

   fraud_detection/data/creditcard.csv

4. The file should be ~150MB with 284,807 rows and these columns:
   Time, V1-V28, Amount, Class
   (Class = 0 for normal, 1 for fraud)

Once the file is in place, run the pipeline from the repo root:

   python fraud_detection/main.py
