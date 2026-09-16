import pandas as pd

excel_path = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\SAMM\SAMM_Micro_FACS_Codes_v2.xlsx'
try:
    df = pd.read_excel(excel_path)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
except Exception as e:
    print(e)
