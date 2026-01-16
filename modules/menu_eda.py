import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Loading Dataset
food_sales = pd.read_csv('/Users/ozdemir/Desktop/Task 2/EU_park_food_sales.csv')

# 1. Analysis: Best Selling Items
# This shows the "Anchor" items that sells the most.
item_totals = food_sales.sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=item_totals.index, y=item_totals.values, palette='mako')
plt.title('Total Sales Volume per Menu Item', fontsize=15)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Items')
plt.ylabel('Units Sold')
plt.tight_layout()
plt.savefig('item_popularity.png')

# 2. Analysis: Basket Size
# This explains why orders have 10-15 items per transactions (Group behavior).
food_sales['OrderSize'] = food_sales.sum(axis=1)

plt.figure(figsize=(10, 5))
sns.histplot(food_sales['OrderSize'], bins=range(5, 25), kde=True, color='#FF9900')
plt.title('Distribution of Items per Order (Basket Size)', fontsize=15)
plt.xlabel('Number of Items in One Transaction')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('basket_size_dist.png')

# 3. Analysis: Correlation Heatmap
# Convert to binary matrix (1 if bought, 0 if not) to find association groups.
food_binary = (food_sales.drop(columns=['OrderSize']) > 0).astype(int)
corr_matrix = food_binary.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='YlGnBu', annot=False)
plt.title('Item Co-occurrence Heatmap: Identifying Combo Groups', fontsize=15)
plt.tight_layout()
plt.savefig('item_group_heatmap.png')

