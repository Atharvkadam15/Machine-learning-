import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.preprocessing import StandardScaler


def MarvellousClassifier(Datapath):
  border = "_"*40

  print(border)
  print("Step 1 : Load the Dataset from CSV file ")
  print(border)

  df = pd.read_csv(Datapath)

  print(border)
  print("Some enteries from Dataset are : ")
  print(df.head())
  print(border)

def main():
  MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
  main()