import pandas as pd

set = pd.read_csv('pexels.csv')

set = set.sample(n=1000)
print(set)

set.to_csv("./pexels_set.csv", sep=',', index=False)

#html = set.to_html()
#with open('./landscapes_bottom_set.html', 'w') as file:
#    file.write(html)
