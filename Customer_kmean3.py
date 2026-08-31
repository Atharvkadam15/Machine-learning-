import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
  # Step 1 : load the data

  df = pd.read_csv("Mall_Customers.csv")
  print("Dataset loaded successfully")
  print(df.head())

  print("missing values : ")
  print(df.isnull().sum())

  # Step 2 : Feature Selection

  X = df[["AnnualIncome" , "SpendingScore"]]
  print("selected features : ")
  print(X.head())

  # Step 3 : Scale the Data

  scalar = StandardScaler()

  X_scaled = scalar.fit_transform(X) #what

  print("Scaled Data : ")
  print(X_scaled[:5])
  

if __name__ == "__main__":
  main()