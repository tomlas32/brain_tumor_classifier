"""
Download a brain tumor MRI classification dataset from Kaggle using KaggleHub.

- Sets a custom download directory using the KAGGLEHUB_CACHE environment variable.
- Creates the directory if it doesn't exist.
- Downloads the dataset from the KaggleHub repository: 'sartajbhuvaji/brain-tumor-classification-mri'.
- Handles potential errors gracefully using try-except blocks.

Requirements:
- kagglehub package
- Internet access
- Kaggle API credentials properly configured

Outputs:
- Path to the downloaded dataset is printed upon success.

Author: Tomasz Lasota
Date: 2025-07-27
Version: 1.0
"""



import kagglehub
import os


# Set custom kaggle data download destination directory
data_dir = "C:\\Users\\tomla\\Documents\\Projects\\brain_tumor_classifier\\data"
os.environ["KAGGLEHUB_CACHE"] = data_dir

os.makedirs(data_dir, exist_ok=True)

try:
    print("Downloading dataset...")
    path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
    print("✅ Dataset downloaded successfully.")
    print("📁 Files are located at:", path)

except kagglehub.exceptions.KaggleHubError as e:
    print("❌ KaggleHub error occurred:", e)

except Exception as e:
    print("❌ An unexpected error occurred:", e)