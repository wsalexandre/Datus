#William S. Alexandre
#Just a kidding! I never won in this lottery, yet! :P

import numpy as np
import pandas as pd
import sys
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Download lotofacil results and put inside current directory  
#http://www1.caixa.gov.br/loterias/_arquivos/loterias/D_lotfac.zip
df_list = pd.read_html('d_lotfac.htm', header=None, encoding="utf8")

#Converting html to CSV
for i, df in enumerate(df_list):
    df.to_csv('lotofacil{}.csv'.format(i))

#Using numpy to load array 
dataset = np.loadtxt('lotofacil0.csv', delimiter=",", usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), skiprows=1)

#Converting np array to a transaction encoder for mlxtend
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

#Defining minimum support to get numbers that came togheter in the same  draw in this case 23%
frequent_itemsets = apriori(df, min_support=0.23, use_colnames=False)

print(frequent_itemsets)

#voilá
frequent_itemsets.to_excel("output.xlsx")
