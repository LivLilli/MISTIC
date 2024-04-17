from datetime import datetime
import pandas as pd
import numpy as np
from setfit import SetFitModel

t1 = datetime.now()

input_data_path = "results/gold_extended.csv"
output_data_path = ''
model_name = ''

df_split = pd.read_csv(input_data_path)[['id', 'gold', 'splitted_text']]
model = SetFitModel.from_pretrained("./" + model_name)
df_split['classification'] = model.predict(df_split.splitted_text)
df_split['classification'] = np.where(df_split['classification'] < 1, 0, 1)
df_split.to_csv(output_data_path, index=False)

t2 = datetime.now()
t2 - t1
