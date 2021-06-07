import pandas as pd

landscapes_data = pd.read_csv('results_landscapes.csv')

top = landscapes_data.nsmallest(500, 'orig_ia_pre_score')

top = top[['image_id', 'orig_ia_pre_score', 'orig_ia_pre_styles_change']]
print(top)

top.to_csv("./landscapes_bottom_set.csv", sep=',', index=False)

html = top.to_html()
with open('./landscapes_bottom_set.html', 'w') as file:
    file.write(html)
