import os

class Config:
        # Define the base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the data directory
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Define the output directory
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    @staticmethod
    def create_output_directory():
        """
        Create the output directory if it doesn't exist.
        """
        if not os.path.exists(Config.OUTPUT_DIR):
            os.makedirs(Config.OUTPUT_DIR)
