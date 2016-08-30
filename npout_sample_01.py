import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

%matplotlib inline

fonts = [font.name for font in fm.fontManager.ttflist if os.path.exists(font.fname) and os.stat(font.fname).st_size>1e6]
mpl.rcParams['font.family'] = 'Microsoft JhengHei'

--
# import json, codecs
# colMap = json.load(codecs.open('D:/python_examples/machine_learning/csv/npout_columns.json', 'r', 'utf-8'))
# print colMap['NP_IN']

df = pd.read_csv('npout2_data_05.csv', encoding='utf-8', sep='|', skiprows=0, names=['C%s' % (i+1) for i in xrange(218)], dtype={'C1': str, 'C3': str})
df[df['C3']=='0916727473']

df2 = pd.read_csv('npout2_view_competitors.csv', encoding='utf-8', sep='|', skiprows=0, names=['C1', 'C3', 'C219', 'C220', 'C221', 'C222'], dtype={'C1': str, 'C3': str})

df_mg = pd.merge(df, df2, how='left', on=['C1', 'C3'])
df_mg[~df_mg['C219'].isnull()]


df_mg.to_csv('npout2_newData_07.csv', sep='|', mode='w', encoding='utf-8', header=False, index=False)

--
df = pd.read_csv('D:/python_examples/machine_learning/csv/npout_sample.csv', encoding='utf-8', sep='\\t', skiprows=1, names='YM,SUBSCTN_ID,SUBSCTN_START_DT,NP_IN,TENURE,DT1_DATA_RTD_DUR,DT1_DATA_RTD_DWN_PACK,DT1_DATA_RTD_UP_PACK,DT1_DATA_RTD_CNT,DT2_DATA_RTD_DUR,DT2_DATA_RTD_DWN_PACK,DT2_DATA_RTD_UP_PACK,DT2_DATA_RTD_CNT,DT3_DATA_RTD_DUR,DT3_DATA_RTD_DWN_PACK,DT3_DATA_RTD_UP_PACK,DT3_DATA_RTD_CNT,VO1_MIN,VO1_CNT,VO1_PTYCNT,VO1_CHTMIN,VO1_TWMMIN,VO1_FETMIN,VO1_CHTCNT,VO1_TWMCNT,VO1_FETCNT,VO1_CHTPTYCNT,VO1_TWMPTYCNT,VO1_FETPTYCNT,VO2_MIN,VO2_CNT,VO2_PTYCNT,VO2_CHTMIN,VO2_TWMMIN,VO2_FETMIN,VO2_CHTCNT,VO2_TWMCNT,VO2_FETCNT,VO2_CHTPTYCNT,VO2_TWMPTYCNT,VO2_FETPTYCNT,VO3_MIN,VO3_CNT,VO3_PTYCNT,VO3_CHTMIN,VO3_TWMMIN,VO3_FETMIN,VO3_CHTCNT,VO3_TWMCNT,VO3_FETCNT,VO3_CHTPTYCNT,VO3_TWMPTYCNT,VO3_FETPTYCNT,SMS1_CNT,SMS2_CNT,SMS3_CNT,TENURE_MONTH_CNT,PREPAID_IND,GENDER_TYPE_CD,SUBSCTN_AGE,NTW_SERVICE_TYPE_CD,OFFER_SERVICE_TYPE_CD,OFFER_ID,BRANCH_CHANNEL_ID,SC_CHANNEL_ID,AGENT_CHANNEL_ID,MPRO_MEMBER_IND,NP_IN_IND,NP_FROM_CARRIER_CD,NP_OUT_IND,NP_TO_CARRIER_CD,HANDSET_RETN_CONTR_IND,HANDSET_CONTR_EXP_MONTH_CNT,HIST_HANDSET_CONTR_CNT,HIST_HANDSET_PURCHASE_CNT,AVG_HANDSET_PURCHASE_AMT,LAST_HANDSET_PURCHASE_AMT,OFFER_2GTO3G_IND,OFFER_3GTO2G_IND,OFFER_2GTO4G_IND,OFFER_3GTO4G_IND,OFFER_4GTO2G_IND,OFFER_4GTO3G_IND,VB1_TOT_PTYCNT,VB1_TOT_CNT,VB1_TOT_DUR,VB1_TWM_PTYCNT,VB1_TWM_CNT,VB1_TWM_DUR,VB1_FET_PTYCNT,VB1_FET_CNT,VB1_FET_DUR,VB1_VBT_PTYCNT,VB1_VBTCNT,VB1_VBTDUR,VB1_APW_PTYCNT,VB1_APW_CNT,VB1_APW_DUR,VB2_TOT_PTYCNT,VB2_TOT_CNT,VB2_TOT_DUR,VB2_TWM_PTYCNT,VB2_TWM_CNT,VB2_TWM_DUR,VB2_FET_PTYCNT,VB2_FET_CNT,VB2_FET_DUR,VB2_VBT_PTYCNT,VB2_VBTCNT,VB2_VBTDUR,VB2_APW_PTYCNT,VB2_APW_CNT,VB2_APW_DUR,VB3_TOT_PTYCNT,VB3_TOT_CNT,VB3_TOT_DUR,VB3_TWM_PTYCNT,VB3_TWM_CNT,VB3_TWM_DUR,VB3_FET_PTYCNT,VB3_FET_CNT,VB3_FET_DUR,VB3_VBT_PTYCNT,VB3_VBTCNT,VB3_VBTDUR,VB3_APW_PTYCNT,VB3_APW_CNT,VB3_APW_DUR'.split(','), na_values='?')
pd.set_option("display.max_columns", 130)

df.head(5)

--
dfc = df.corr()
indices = np.where(dfc > 0.7)
cols = [(dfc.index[x], dfc.columns[y]) for x, y in zip(*indices) if x != y and x < y]
cols
# for x,y in cols:
  # if y in df.index:
    # del df[y]
# df['TENURE'].corr(df['TENURE_MONTH_CNT'])

--
# np_in 遠傳, 台灣之星, 台哥大, 亞太
df["Target"] = df['NP_IN'].replace({name: n for n, name in enumerate(df['NP_IN'].unique())}) #coding

# plot
df['NP_IN'].value_counts().plot(kind='pie')
df['TENURE'].value_counts().sort_index().plot(kind='bar')
df['PTY_GENDER'].value_counts().plot(kind='bar')
df['OFFER_ID'].value_counts().plot(kind='bar')
df.boxplot(column=['DT1_DATA_RTD_DUR'])

plt.show()

# transform
cols = list(df.columns)

df['DT1_DATA_RTD_DUR'].dropna().describe()
df['DT1_DATA_RTD_DUR'].dropna().count()

df['DT1_DATA_RTD_DUR_L'] = df['DT1_DATA_RTD_DUR'].map(np.log)
df['DT1_DATA_RTD_DUR_L'].fillna(0, inplace=True)

df['DT1_DATA_RTD_DUR_Z'] = (df['DT1_DATA_RTD_DUR_L'] - df['DT1_DATA_RTD_DUR_L'].mean())/df['DT1_DATA_RTD_DUR_L'].std(ddof=0)

# Binning（Percentile To Z-Score：33%=-0.4399、66%=0.4125）
# https://www.easycalculation.com/statistics/percentile-to-z-score.php
def binning(col, cut_points):
  #create list by adding min and max to cut_points
  colBin = pd.cut(col,bins=([col.min()]+cut_points+[col.max()]),labels=range(len(cut_points)+1),include_lowest=True)
  return colBin

df['DT1_DATA_RTD_DUR_C'] = binning(df['DT1_DATA_RTD_DUR_Z'], [-0.4399, 0.4125])

df['TENURE_Z'] = (df['TENURE'] - df['TENURE'].mean())/df['TENURE'].std(ddof=0)
df['TENURE_C'] = binning(df['TENURE_Z'], [-0.4399, 0.4125])

# df['DT1_DATA_RTD_DUR'].fillna(df['DT1_DATA_RTD_DUR'].dropna().mean(), inplace=True)
# df['DT1_DATA_RTD_DUR'].fillna(df['DT1_DATA_RTD_DUR'].dropna().median(), inplace=True)
# df['DT1_DATA_RTD_DUR'].fillna(df['DT1_DATA_RTD_DUR'].dropna().mode()[0], inplace=True)
# df['DT1_DATA_RTD_DUR'].fillna(df.groupby("OFFER_ID")["DT1_DATA_RTD_DUR"].transform("mean"), inplace=True)
# df["DT1_DATA_RTD_DUR_NEW"] = df.groupby(["OFFER_ID"]).transform(lambda x: x["DT1_DATA_RTD_DUR"].fillna(x.["DT1_DATA_RTD_DUR"].dropna().mean()))

--
pt = pd.pivot_table(df, values='YM', index=['NP_IN','NP_FROM_CARRIER_CD'], aggfunc=len)
key1 = pt.index.labels[0]
key2 = pt.rank(ascending=False)
sorted_pt = pt.take(np.lexsort((key2, key1)))
sorted_pt

--

from sklearn import tree

X = df[['TENURE_C', 'DT1_DATA_RTD_DUR_C']]
Y = df["Target"]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=df[['TENURE_C', 'DT1_DATA_RTD_DUR_C']].columns, class_names=df['NP_IN'].unique(), filled=True, rounded=True, special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
Image(graph.create_png())

def get_code(tree, feature_names):
  left = tree.tree_.children_left
  right = tree.tree_.children_right
  threshold = tree.tree_.threshold
  features = [feature_names[i] for i in tree.tree_.feature]
  value = tree.tree_.value
  def recurse(left, right, threshold, features, node):
    if threshold[node] != -2:
      print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
      if left[node] != -1:
        recurse (left, right, threshold, features,left[node])
      print "} else {"
      if right[node] != -1:
        recurse (left, right, threshold, features,right[node])
      print "}"
    else:
      print "return " + str(value[node])
  recurse(left, right, threshold, features, 0)

