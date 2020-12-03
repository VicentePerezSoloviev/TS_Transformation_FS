
# loading essential libraries first
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# import data
mdata = sm.datasets.macrodata.load_pandas().data
df = mdata.iloc[:, 2:12]
df.head()

from TS_transformations import TSTransformations

TSTransf = TSTransformations(df)
transformations = ['detrend', 'smooth', 'log']
variables = list(df.columns)
variable_y = 'pop'
variables = list(set(variables) - {variable_y})

for var in variables:
    transformation = TSTransf.de_trending(var)
    df[var + 'detrend'] = transformation

for var in variables:
    transformation = TSTransf.smoothing(var, window=10)
    df[var + 'smooth'] = transformation

for var in variables:
    transformation = TSTransf.log(var)
    df[var + 'log'] = transformation


def cost_function(variables_list):
    nobs, maxlags, forecastings = 20, 15, 10
    data = df[variables_list + [variable_y]]

    df_train, df_test = data[0:-nobs], data[-nobs:]

    model = VAR(df_train)
    results = model.fit(maxlags=maxlags, ic='aic')

    lag_order = results.k_ar
    array = results.forecast(df_train.values[-lag_order:], forecastings)

    variables_ = list(data.columns)
    position = variables_.index(variable_y)

    validation = [array[i][position] for i in range(len(array))]
    mae = mean_absolute_error(validation, df_test['pop'][-forecastings:])
    return mae


vector = pd.DataFrame(columns=list(variables))
vector.loc[0] = 0.5

from TransformationsFeatureSelection import TransformationsFSEDA as EDA
eda = EDA(max_it=10, dead_it=10, size_gen=10, alpha=0.7, vector=vector,
          array_transformations=transformations, cost_function=cost_function)
best_ind, best_MAE = eda.run(output=True)

print(best_ind, best_MAE)

plt.figure()
hist = eda.historic_best
plt.plot(list(range(len(hist))), hist)
plt.show()




