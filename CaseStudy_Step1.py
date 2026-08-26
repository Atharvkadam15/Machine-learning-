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