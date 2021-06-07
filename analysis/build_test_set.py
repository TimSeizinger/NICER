import pandas as pd

df = pd.read_csv('ava.csv')

df = df.sample(n=1000)

df.to_csv('./rating_test_set.csv', sep=',', index=False)

print(True)