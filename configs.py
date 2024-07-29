import os


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define data directories
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    VAL_DIR = os.path.join(BASE_DIR, 'evaluatioan_results')

    # Ensure directories exist
    @staticmethod
    def ensure_directories():
        required_dirs = [
            Config.VAL_DIR
        ]
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)


# Run this script to ensure directories are created
if __name__ == "__main__":
    Config.ensure_directories()
    print("All necessary directories exist.")