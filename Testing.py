import pandas as pd
from pathlib import Path

strlist = []
strlist.append(f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/ssmtpiaa_sgd/fivek-1008_ssmtpiaa_sgd.jpg" style="max-width: 200px; max-height: 200px">'
               f'<p> rating here </p>')
df = pd.DataFrame({'images': strlist})

with open('analysis/sets/survey_results/test.html', 'w') as file:
    html = df.to_html(escape=False)
    file.write(html)