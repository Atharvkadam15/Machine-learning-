import numpy as np
import pandas as pd
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# Step 1 : Load the Data
#-------------------------------------------------------
#    Function Name : LoadData
#.   Description :   Load the data from csv
#.   Input :         Name of the csv file
#.   Output :        Data frame
#.   Author :        Atharv Pradeep Kadam
#.   Date :          16/08/2026
#-------------------------------------------------------

def LoadData(filename):
  df = pd.read_csv(filename)

  print("Dataset loaded successsfully..")
  print(df.head())

  return df


def main():
  LoadData("marvellousTitanicDataset.csv")

if __name__ == "__main__":
  main()