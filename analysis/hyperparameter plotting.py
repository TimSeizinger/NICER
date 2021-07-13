import seaborn as sns
import pandas as pd

df = pd.read_csv('mean_distances.csv')
mean_distances = df.to_dict()

print(mean_distances)