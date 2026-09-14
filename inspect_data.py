import os
import pandas as pd

file_path = "data/customer_support_tickets.csv"

if not os.path.exists(file_path):
    print("Dataset file not found:", file_path)
    exit()

df = pd.read_csv(file_path)

print("Number of tickets:", len(df))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 tickets:")
print(df.head())