import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Define the Data
# (Update these numbers based on the results from your report's calculations)
data = {
    'Country': ['United States', 'United Kingdom', 'Germany', 'Australia'],
    'Estimated Annual Revenue per Store ($ USD)': [85000, 62000, 58000, 71000],
    'Market Saturation Level': ['Low (High Need)', 'High', 'Very High', 'Medium']
}

df = pd.DataFrame(data)

# 2. Set up the plot aesthetics
plt.figure(figsize=(10, 6))
colors = ['#1f77b4' if country == 'United States' else '#cccccc' for country in df['Country']]

# 3. Create the bar chart
bars = plt.bar(df['Country'], df['Estimated Annual Revenue per Store ($ USD)'], color=colors, edgecolor='black')

# 4. Add labels and titles
plt.title('Estimated Annual Revenue Potential per Supermarket by Country', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Annual Revenue ($ USD)', fontsize=12)
plt.xlabel('Analyzed Markets', fontsize=12)
plt.ylim(0, max(df['Estimated Annual Revenue per Store ($ USD)']) * 1.2) # Add headroom for labels

# 5. Add data labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1500, f'${yval:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 6. Add a custom legend/note about the USA selection
plt.axhline(y=0, color='black', linewidth=1.5)
plt.text(0.02, 0.95, '*USA selected due to highest revenue potential\ncombined with low market saturation.', 
         transform=plt.gca().transAxes, fontsize=10, style='italic', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# 7. Display the chart
plt.tight_layout()
plt.show()