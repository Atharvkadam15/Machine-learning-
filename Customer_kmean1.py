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

if __name__ == "__main__":
  main()