import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

man_height = stats.norm.rvs(loc=170, scale=10, size=500, random_state=1)
woman_height = stats.norm.rvs(loc=150, scale=10, size=500, random_state=1)

X = np.concatenate([man_height, woman_height])
Y = ["man"] * len(man_height) + ["woman"] * len(woman_height)

df = pd.DataFrame(list(zip(X,Y)), columns = ["X", "Y"])
fig = sns.displot(data = df, x="X", hue="Y", kind="kde")
fig.set_axis_labels("cm","count")
plt.show()

statistic, pvalue = stats.ttest_ind(man_height, woman_height, equal_var=True)
print("statistics:", statistic)
print("pvalue:", pvalue)
print("*:", pvalue<0.05)
print("**:",pvalue<0.001)