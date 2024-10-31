import os


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define directories
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    STAT_DIR = os.path.join(BASE_DIR, 'general_statistical_llm_based_metrics')

# Run this script to ensure directories are created
if __name__ == "__main__":
    print("All necessary directories exist.")