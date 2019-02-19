#Association Analysis
# do now know which tool to use
# Assiciation analysis can help find the common patterns of items in large data
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df.head()

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset = ['InvoiceNo'], inplace = True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
df.head()

# one hot-encoded: categorical
#

basket = (df[df['Country']=="France"].groupby(['InvoiceNo','Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo') )
basket.head()

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

# refrequent items
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
# Association
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
# get pairs with high lift and confidence
rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]


##
