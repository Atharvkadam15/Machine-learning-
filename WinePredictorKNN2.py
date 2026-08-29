import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import StandardScaler


def MarvellousClassifier(Datapath):
  border = "_"*40


  # Step 1 : Load the Dataset from CSV file
  print(border)
  print("Step 1 : Load the Dataset from CSV file ")
  print(border)

  df = pd.read_csv(Datapath)

  print(border)
  print("Some enteries from Dataset are : ")
  print(df.head())
  print(border)

  # Step 2 : Clean the Dataset  
  print(border)
  print("Step 2 : Clean the Dataset ")
  print(border)

  df.dropna(inplace = True)

  print("Shape of Dataset : ",df.shape)
  print("Total records : ", df.shape[0])
  print("Total columnsn : ", df.shape[1])

  print(border)

  


def main():
  MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
  main()