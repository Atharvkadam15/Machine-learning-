import pandas as pd

Border = "-"*30

###########################################
# Step 1 : Load the Dataset 
###########################################

print(Border)
print("Step 1 : Load the Dataset")
print(Border)

DataPath = "iris.csv"

df = pd.read_csv(DataPath)

print("Dataset loaded successfully")
print("Initial entries from dataset are : ")

print(df.head())

###########################################
# Step 2 : Data Analysis (EDA) 
###########################################

print(Border)
print("Step 2 : Data Analysis (EDA) ")
print(Border)

print("Shape of Dataset : ",df.shape)

print("Column Names : ",list(df.columns))

print("Missing Values per Column : ")
print(df.isnull().sum())

print("Class Distribution (species count)")
print(df["species"].value_counts())

print("Statistical report of dataset : ")
print(df.describe())

