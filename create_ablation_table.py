import pandas as pd

# Define data based on the read files
# Removed 'F1-Score' as it is identical to 'UF1'
data = [
    {
        "Method": "ResNet-18 (RGB only)",
        "Accuracy": "0.6966 ± 0.0577",
        "UAR": "0.6496 ± 0.0271",
        "UF1": "0.6323 ± 0.0495"
    },
    {
        "Method": "ResNet-18 (Dynamic Image only)",
        "Accuracy": "0.8590 ± 0.0138",
        "UAR": "0.8612 ± 0.0160",
        "UF1": "0.8393 ± 0.0170"
    },
    {
        "Method": "ResNet-18 (RGB + Dynamic Fusion)",
        "Accuracy": "0.8761 ± 0.0060",
        "UAR": "0.8463 ± 0.0134",
        "UF1": "0.8520 ± 0.0077"
    },
    {
        "Method": "+ SPP (Spatial Pyramid Pooling)",
        "Accuracy": "0.9038 ± 0.0138",
        "UAR": "0.8679 ± 0.0105",
        "UF1": "0.8822 ± 0.0207"
    },
    {
        "Method": "+ Transfer Learning (JAFFE)",
        "Accuracy": "0.9103 ± 0.0138",
        "UAR": "0.8787 ± 0.0235",
        "UF1": "0.8898 ± 0.0156"
    }
]

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV with 'utf-8-sig' encoding
csv_path = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\CASME2_Ablation_Study.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Table updated and saved to {csv_path} (F1-Score column removed)")
print(df)
