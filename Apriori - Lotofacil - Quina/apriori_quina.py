#William S. Alexandre
#Just a kidding! I never won in this lottery, Brazillian Quina's select five numbers on 80, yet!

import pandas as pd
import sys
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Download lotofacil results and put inside current directory
#http://www1.caixa.gov.br/loterias/_arquivos/loterias/D_quina.zip
df_list = pd.read_html('d_quina.htm', header=None, encoding="utf8")

#Converting html to CSV
for i, df in enumerate(df_list):
    df.to_csv('quina{}.csv'.format(i))

#Using numpy to load array
dataset = np.loadtxt('quina0.csv', delimiter=",",
                     usecols=(3, 4, 5, 6, 7), skiprows=1)


#Converting np array to a transaction encoder for mlxtend
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

#Defining minimum support to get numbers that came togheter in the same  draw in this case only in 10%
frequent_itemsets = apriori(df, min_support=0.10, use_colnames=False)

print(frequent_itemsets)

frequent_itemsets.to_excel("output_quina.xlsx")


