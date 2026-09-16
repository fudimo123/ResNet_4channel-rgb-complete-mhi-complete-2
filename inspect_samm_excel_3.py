import pandas as pd

excel_path = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\SAMM\SAMM_Micro_FACS_Codes_v2.xlsx'
try:
    df = pd.read_excel(excel_path, header=13) # Header at row 14 (index 13)
    print("Columns:", df.columns.tolist())
    print("First 3 rows:")
    print(df.head(3))
except Exception as e:
    print(e)
