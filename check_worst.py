import pandas as pd

df = pd.read_excel('evaluation_output/evaluation_results.xlsx')
df_sorted = df.sort_values('MRE_mm', ascending=False)

print('TOP 10 WORST PERFORMING IMAGES:')
print(df_sorted.head(10).to_string(index=False))
print()
print('TOP 10 BEST PERFORMING IMAGES:')
print(df_sorted.tail(10).to_string(index=False))
print()
print(f'Mean MRE:              {df["MRE_mm"].mean():.3f}')
print(f'Median MRE:            {df["MRE_mm"].median():.3f}')
print(f'Images with MRE > 3mm: {(df["MRE_mm"] > 3).sum()}')
print(f'Images with MRE > 5mm: {(df["MRE_mm"] > 5).sum()}')