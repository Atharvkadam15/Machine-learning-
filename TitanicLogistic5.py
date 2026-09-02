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

# Step 3 : Split Data
#-------------------------------------------------------
#    Function Name : Split Data
#.   Description :   it performs Spliting activity
#.   Input :         Dataframe
#.   Output :        4 subsets for training and testing
#.   Author :        Atharv Pradeep Kadam
#.   Date :          16/08/2026
#-------------------------------------------------------

def SplitData(df):
  X = df.drop("Survived",axis = 1)
  Y = df["Survived"]

  X_train,X_test,Y_train,Y_test= train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
  )

  print("Data Split Successfully..")

  return X_train, X_test, Y_train, Y_test


# Step 4 : Train Model
#-------------------------------------------------------
#    Function Name : Train Model
#.   Description :   it performs Model training
#.   Input :         Training features and labels
#.   Output :        Trained Model
#.   Author :        Atharv Pradeep Kadam
#.   Date :          16/08/2026
#-------------------------------------------------------

def TrainModel(X_train,Y_train):
  model = LogisticRegression(max_iter=1000)

  model = model.fit(X_train,Y_train)
  print("Model Trained Successfully..")

  return model

# Step 5 : Evalute Model
#-------------------------------------------------------
#    Function Name : Evalute Model
#.   Description :   It performs Model testing
#.   Input :         model, testing data (features , labels)
#.   Output :        
#.   Author :        Atharv Pradeep Kadam
#.   Date :          16/08/2026
#-------------------------------------------------------

def EvaluateModel(model, X_test, Y_test):
  Y_pred = model.predict(X_test)

  accuracy = accuracy_score(Y_test,Y_pred)

  print("Accuracy : ",accuracy)

  print()

  # Step 5 : Model Preservation
#-------------------------------------------------------
#    Function Name : Model Preservation
#.   Description :   It performs Model Preservation
#.   Input :         model
#.   Output :        
#.   Author :        Atharv Pradeep Kadam
#.   Date :          16/08/2026
#-------------------------------------------------------

def PreserveModel(model,filename):
  joblib.dump(model,filename)

  print("Model preserved with Name : ",filename)
  
def main():
  #Step 1
  df = LoadData("marvellousTitanicDataset.csv")

  #Step 2
  df = preprocessing(df)

  # Step 3
  X_train, X_test, Y_train, Y_test = SplitData(df)

  # Step 4 
  model = TrainModel(X_train,Y_train)

  # Step 5 
  EvaluateModel(model,X_test,Y_test)

  # Step 6 
  PreserveModel(model,"MarvellousTitanic.pkl")

if __name__ == "__main__":
  main()