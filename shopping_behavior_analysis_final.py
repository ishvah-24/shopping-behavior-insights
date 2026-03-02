# ===============================
# IMPORTS
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import chi2_contingency
from catboost import CatBoostRegressor, Pool

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('shopping_behavior_updated (1).csv')

# ===============================
# CLEAN COLUMN NAMES (minimal fix so underscores work later)
# ===============================
df.columns = df.columns.str.replace(' ', '_').str.replace('(', '', regex=False).str.replace(')', '', regex=False)

# ===============================
# BASIC DATA UNDERSTANDING
# ===============================
print(df.head())
print(df.describe())
print(df.shape)
print(df.columns)
print(df.dtypes)
print("Sum of duplicates:", df.duplicated().sum())
print("NaN values:", df.isna().sum())
print("NULL values:", df.isnull().sum())

# ===============================
# UNIVARIATE ANALYSIS
# ===============================
print("Age Min:", df['Age'].min(),
      "Max:", df['Age'].max(),
      "Mean:", df['Age'].mean(),
      "Std:", df['Age'].std())

print("Purchase Amount Mean:", df['Purchase_Amount_USD'].mean(),
      "Median:", df['Purchase_Amount_USD'].median(),
      "Review Rating Mean:", df['Review_Rating'].mean(),
      "Previous Purchases Mean:", df['Previous_Purchases'].mean(),
      "Median:", df['Previous_Purchases'].median())

sns.histplot(df['Age'], kde=True, bins=30)
plt.title("Age Distribution")
plt.show()

sns.histplot(df['Purchase_Amount_USD'], kde=True, bins=30)
plt.title("Purchase Amount Distribution")
plt.show()

sns.histplot(df['Review_Rating'], kde=True, bins=30)
plt.title("Review Rating Distribution")
plt.show()

sns.histplot(df['Previous_Purchases'], kde=True, bins=30)
plt.title("Previous Purchases Distribution")
plt.show()

# ===============================
# FILTERING (Summer & Spring Clothing Revenue)
# ===============================
summer_subset = df[(df['Season']=='Summer') & (df['Category']=='Clothing')]
summer_revenue = summer_subset['Purchase_Amount_USD'].sum()
print("Summer Clothing Revenue:", summer_revenue)

spring_subset = df[(df['Season']=='Spring') & (df['Category']=='Clothing')]
spring_revenue = spring_subset['Purchase_Amount_USD'].sum()
print("Spring Clothing Revenue:", spring_revenue)

payment_method_revenue_sum = df.groupby('Payment_Method')['Purchase_Amount_USD'].sum().reset_index()
print("Payment Method Revenue Sum:",payment_method_revenue_sum)

# ===============================
# SIZE ANALYSIS
# ===============================
df['Size'] = pd.Categorical(df['Size'], categories=['S','M','L','XL'], ordered=True)
state_size_counts = df.groupby('Location')['Size'].value_counts().reset_index(name='count')
print(state_size_counts)

sns.barplot(data=state_size_counts,  y='Location',  x='count',  hue='Size',  palette='viridis', errorbar=None)
plt.title("Size Distribution per Location")
plt.show()

state_size_table = state_size_counts.pivot_table(columns='Size', values='count', index='Location')
print(state_size_table)

plot = sns.countplot(data=df,  x= 'Gender',  hue='Size',  palette='flare')
plot.set_title('Item size to Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend()
plt.show()

# Create contingency table
gender_size = pd.crosstab(df['Gender'], df['Size'])
# Run Chi-square test
chi2, p, dof, expected = chi2_contingency(gender_size)
print("Gender vs Size")
print("Chi-square:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)

location_size = pd.crosstab(df['Location'], df['Size'])
chi2, p, dof, expected = chi2_contingency(location_size)
print("Location vs Size")
print("Chi-square:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print(gender_size.div(gender_size.sum(axis=1), axis=0))

# ===============================
# CUSTOMER SEGMENTATION
# ===============================
age_bins = [17, 25, 35, 45, 55, 65, 70]
age_labels = ['18-25','26-35','36-45','46-55','56-65','66+']
df['Age_Groups'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
print(df['Age_Groups'].value_counts())

loyalty_bins = [0, 10, 25, 50]
loyalty_labels = ['Low','Medium','High']
df['Loyalty_Level'] = pd.cut(df['Previous_Purchases'], bins=loyalty_bins, labels=loyalty_labels)
print(df['Payment_Method'].value_counts())
print(df['Gender'].value_counts())

# ===============================
# REVENUE ANALYSIS
# ===============================
loyalty_revenue = df.groupby('Loyalty_Level')['Purchase_Amount_USD'].sum()
print("Revenue per Loyalty Level:", loyalty_revenue)

age_group_revenue = df.groupby('Age_Groups')['Purchase_Amount_USD'].sum()
print("Revenue per Age Group:", age_group_revenue)

plot = sns.barplot(data=df,  y='Location',  x='Purchase_Amount_USD',  estimator=np.sum, errorbar=None,  hue='Location',  palette='magma')
plt.title("Revenue per Location")
plt.xlabel("Total Revenue (USD)")
plt.ylabel("Location")
plt.show()

plot = sns.barplot(data=df,  x='Season',  y='Purchase_Amount_USD',  hue='Category',  estimator=np.sum, errorbar=None,  palette='viridis')
plt.title("Category Revenue per Season")
plt.show()

plot = sns.countplot(data=df,  x='Discount_Applied',  hue='Discount_Applied',  palette='flare')
plt.title("Discount Applied Counts")
plt.show()

plot = sns.histplot(data=df,  x='Review_Rating',  hue='Discount_Applied',  multiple='dodge',  bins=10)
plt.title("Review Ratings by Discount")
plt.show()

plot = sns.boxplot(data=df,  x='Discount_Applied',  y='Review_Rating',  palette='flare')
plt.title("Discount Data Spread according to Rating Review")
plt.show()

plot = sns.boxplot(data=df,  y='Subscription_Status',  x='Previous_Purchases')
plt.title("Previous Purchases vs Subscription Status")
plt.xlabel("Number of Previous Purchases")
plt.ylabel("Subscription Status")
plt.show()

plot = sns.countplot(data=df,  x='Age_Groups',  hue='Size',  palette='flare')
plot.set_title('Item size to Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.show()

plot = sns.barplot(data=df,  x='Payment_Method',  y='Purchase_Amount_USD',  estimator=np.sum, errorbar=None,  hue='Payment_Method',  palette='flare' )
plt.title('Payment Method revenue generation')
plt.xlabel('Payment Method')
plt.ylabel('Total (USD)')
plt.show()

# ===============================
# ANOVA
# ===============================
model_anova = ols('Purchase_Amount_USD ~ C(Age_Groups)', data=df).fit()
anova_table = sm.stats.anova_lm(model_anova, type=1)
print(anova_table)
eta_sq = anova_table.loc['C(Age_Groups)', 'sum_sq'] / anova_table['sum_sq'].sum()
print("Eta Squared:", eta_sq)

# ===============================
# T-TEST (minimal: match report, compare Previous_Purchases)
# ===============================
if df['Subscription_Status'].dtype != object:
    df['Subscription_Status'] = df['Subscription_Status'].map({1:'Yes',0:'No'})

subscribed = df[df['Subscription_Status']=='Yes']['Previous_Purchases']
not_subscribed = df[df['Subscription_Status']=='No']['Previous_Purchases']

t_stat, p_value = stats.ttest_ind(subscribed, not_subscribed, equal_var=False)
print("T-stat:", t_stat)
print("P-value:", p_value)
print("Subscribed Mean:", subscribed.mean())
print("Not Subscribed Mean:", not_subscribed.mean())

# ===============================
# FEATURE ENGINEERING
# ===============================
binary_cols = ['Subscription_Status', 'Discount_Applied']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['log_purchase'] = np.log1p(df['Purchase_Amount_USD'])

features = [
 'Age', 'Gender', 'Item_Purchased', 'Category', 'Size', 'Color',
 'Season', 'Subscription_Status', 'Discount_Applied',
 'Previous_Purchases', 'Payment_Method', 'Frequency_of_Purchases'
]

target = 'log_purchase'
X = df[features]
y = df[target]

cat_features = ['Gender', 'Item_Purchased', 'Category', 'Size', 'Color', 'Season', 'Payment_Method', 'Frequency_of_Purchases']

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# TRAIN CATBOOST REGRESSOR
# ===============================
model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, eval_metric='MAE', random_seed=42, verbose=100)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# ===============================
# PREDICTIONS
# ===============================
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# ===============================
# EVALUATION
# ===============================
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.3f}")
