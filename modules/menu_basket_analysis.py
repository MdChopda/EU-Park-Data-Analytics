import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load your dataset
food_sales = pd.read_csv('/Users/ozdemir/Desktop/Task 2/EU_park_food_sales.csv')

# 2. Transforming Dataset into Binary Format is Required for the Algorihm
df_binary = (food_sales > 0).astype(bool)

# 3. Applying Apriori Algorithm
# min_support=0.1 means the combo must appear in at least 10% of all transactions
frequent_itemsets = apriori(df_binary, min_support=0.1, use_colnames=True)

# 4. Generating Association Rules
# We used 'lift' as the primary metric to find groups that are NOT random
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)

# 5. Sorting by Lift to find the strongest "Group Structures"
top_rules = rules.sort_values('lift', ascending=False).head(100)

# Displaying Top Rules
print("--- Top Recommended Menu Combinations for Gold Pass ---")
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
