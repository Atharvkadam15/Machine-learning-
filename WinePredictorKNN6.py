import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

def MarvellousClassifier (Datapath):
    border = "-"*60

    # Step 1:Load the Dataset from CSV File

    print(border)
    print("Step 1:Load the Dataset from CSV File")
    print(border)

    df = pd.read_csv(Datapath)

    print(border)
    print("Some Entries From Dataset:")
    print(df.head())
    print(border)

    # Step 2: Clean the Dataset

    print(border)
    print("Step 2: Clean the Dataset")
    print(border)

    df.dropna(inplace=True)
    print("Shape of Dataset :",df.shape)
    print("Total Records :",df.shape[0])
    print("Total Coloumns :",df.shape[1])

    print(border)

    #Step 3 : Separate Independant and Dependant Variables

    print(border)
    print("Step 3: Separate Independant and Dependant Variables")
    print(border)   

    X = df.drop(columns=["Class"])
    Y = df["Class"]

    print("Shape of X :",X.shape)
    print("Shape of Y :",Y.shape)

    print(border)
    print("Inpoy Coloumns :",X.columns.tolist())
    print("Output Coloumn : Class")
    print(border)

    #Step 4 : Split the dataset for training and testing

    print(border)
    print("Step 4 : Split the dataset for training and testing")
    print(border) 

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.5, random_state=42, stratify=Y)
    print(border)
    print("Details of training and testing")

    print("Shape of X_train :",X_train.shape)
    print("Shape of X_est :",X_test.shape)

    print("Shape of Y_train :",Y_train.shape)
    print("Shape of Y_train :",Y_test.shape)

    print(border)

    #Step 5 : Feature Scaling

    print(border)
    print("Step 5 : Feature Scaling")
    print(border)

    scalar = StandardScaler()
    X_train_scaled =scalar.fit_transform(X_train)
    X_test_scaled =scalar.fit_transform(X_test)

    print("Featurer Scaling Done")

    print(border)

    # Step 6 : Build the Model 
    print(border)
    print("Step 6 : Build the Model")
    print(border)   

    model = KNeighborsClassifier(n_neighbors=5)

    print("Classification model is created")

    # Step 7 : Train the Model 
    print(border)
    print("Step 8 : Train the Model")
    print(border)   

    model = model.fit(X_train_scaled,Y_train) 
 
    print("Model Training completed")

    print(border)

    # Step 8 : Test the Model 
    print(border)
    print("Step 8 : Test the Model")
    print(border)     

    Y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(Y_test,Y_pred)

    print("Accuracy : ",accuracy*100)

    
def main():
    MarvellousClassifier("WinePredictor.csv")


if __name__ == "__main__":
    main()