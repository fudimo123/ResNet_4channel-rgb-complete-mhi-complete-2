import pandas as pd

excel_path = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\SAMM\SAMM_Micro_FACS_Codes_v2.xlsx'
try:
    df = pd.read_excel(excel_path, header=None) # Read without header to see raw
    print(df.iloc[10:20])
except Exception as e:
    print(e)
