# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:44:11 2025

@author: Shiri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "C:/Users/Shiri/Downloads/Compressed/online+retail/Online Retail.xlsx"
df = pd.read_excel(file_path, sheet_name="Online Retail")

# Data Cleaning
df = df[df['CustomerID'].notnull()]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['CustomerID'] = df['CustomerID'].astype(int)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# RFM Analysis
current_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency (unique invoices)
    'TotalPrice': 'sum'  # Monetary
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Handling outliers
rfm = rfm[(rfm['Monetary'] > 0) & (rfm['Frequency'] > 0)]

# Scaling the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Clustering using K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], palette='viridis')
plt.title("Customer Segmentation (Recency vs Monetary)")
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.show()
