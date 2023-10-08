import pandas as pd
import matplotlib.pyplot as plt



# Load the uploaded CSV file
chlorella_df = pd.read_csv("chlorella_data.csv")

# Compute descriptive statistics for the Chlorella sp. data
chlorella_descriptive_stats = chlorella_df.describe()
print(chlorella_descriptive_stats)

'''
# Chlorella sp. Before Treatment:
## Mean: 0.3578
## Standard Deviation: 0.4312
## Min: 0.055
## Median (50% percentile): 0.187
## Max: 1.32

# Chlorella sp. After Treatment:
## Mean: 0.4803
## Standard Deviation: 0.4718
## Min: 0.076
## Median: 0.323
## Max: 1.462

# Change in Growth Rate:
## Mean Change: 0.1225
## Standard Deviation of Change: 0.0790
## Min Change: 0.021
## Median Change: 0.1295
## Max Change: 0.271
'''

# Visualizing the growth rates for Chlorella sp. before and after treatment
plt.figure(figsize=(12, 6))

# Chlorella sp.
plt.plot(chlorella_df["Day"], chlorella_df["Chlorella Before"], marker='o', label="Before Treatment", color='blue')
plt.plot(chlorella_df["Day"], chlorella_df["Chlorella After"], marker='o', linestyle='--', label="After Treatment", color='blue')

# Title and labels
plt.title("Growth Rates of Chlorella sp. Before and After Treatment")
plt.xlabel("Day")
plt.ylabel("Growth Rate")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()


