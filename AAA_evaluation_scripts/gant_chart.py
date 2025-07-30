import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

# Define the revised project timeline data
data_updated = {
    'Task': [
        'Planning & Literature Review', 
        'Models set up & Exploratory Analysis', 
        'Inference (create saliency maps and distinctiveness)', 
        'Experiments & Performance Evaluation', 
        'Inference II (experiments comparison)',
        'Thesis Writing & Documentation', 
        'Defense prep'
    ],
    'Start': [
        '2025-02-24', 
        '2025-03-17', 
        '2025-04-15', 
        '2025-05-31',
        '2025-08-15', 
        '2025-09-15', 
        '2025-11-24'
    ],
    'End': [
        '2025-03-30', 
        '2025-05-04', 
        '2025-06-30', 
        '2025-10-01',
        '2025-10-26', 
        '2025-11-23', 
        '2025-12-12'
    ]
}

# Convert to DataFrame
df_updated = pd.DataFrame(data_updated)
df_updated['Start'] = pd.to_datetime(df_updated['Start'])
df_updated['End'] = pd.to_datetime(df_updated['End'])
df_updated['Duration'] = df_updated['End'] - df_updated['Start']

# Plot revised Gantt chart
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(df_updated['Task'], df_updated['Duration'].dt.days, 
         left=df_updated['Start'].map(pd.Timestamp.toordinal), color='lightgreen')

# Format x-axis to show dates
ax.set_xticks(pd.date_range(start=df_updated['Start'].min(), 
                            end=df_updated['End'].max(), freq='M').map(pd.Timestamp.toordinal))
ax.set_xticklabels(pd.date_range(start=df_updated['Start'].min(), 
                                 end=df_updated['End'].max(), freq='M').strftime('%d-%b'), rotation=45)

# Titles and labels
ax.set_title('Master’s Thesis Gantt Chart (24 Feb - 08 Dec, 2025)', fontsize=16)
ax.set_xlabel('Timeline', fontsize=12)
ax.set_ylabel('Tasks', fontsize=12)
ax.grid(True)

plt.tight_layout()
plt.show()
fig.savefig(fname="/home/fkirchhofer/repo/xai_thesis/AAA_evaluation_scripts/Planning_gant_chart_v3.0.png")

# Export the revised timeline to an Excel file
#output_file_path = 'thesis_timeline.xlsx'
#df_updated.to_excel(output_file_path, index=False, engine='openpyxl')
