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


# Step 2 : Data preprocessing
#-------------------------------------------------------
#    Function Name : Preprocessing
#.   Description :   it performs data analysis
#.   Input :         Dataframe
#.   Output :        Updated Dataframe
#.   Author :        Atharv Pradeep Kadam
#.   Date :          16/08/2026
#-------------------------------------------------------

def preprocessing(df):
  df = df.drop([
    "Passengerid",
    "zero"
  ],
  errors = "ignore"
  )

  # Handle missing values 
  df["Age"] = df["Age"].fillna(df["Age"].median())
  df["Fare"] = df["Fare"].fillna(df["Fare"].median())

  df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

  print(df.head())
  print("Data preprocessing Completed..")

  #Converting categorical to numerical
  df = pd.get_dummies(
    df, 
    columns=["Embarked"],
    drop_first=True,
    dtype=int
  )

  return df
  
def main():
  #Step 1
  df = LoadData("marvellousTitanicDataset.csv")

  #Step 2
  df = preprocessing(df)

if __name__ == "__main__":
  main()