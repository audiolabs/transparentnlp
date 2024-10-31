# PrivacyQA Offline Analysis

This repository contains the refactored code from the master branch. Each sub-directory contains data and 
scripts to get the analysis results.

The step to run each sub.directory is as follows:

1. data_processing 
2. general_statistical_llm_based_metrics
3. percentage_comparison

There is an individual README.md file inside each sub-directory.

## Running the Code

To run the code, follow these steps:

. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - **On Windows:**

        ```bash
        venv\Scripts\activate
        ```

    - **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

**Notes**:
You will be prompted to enter you OpenAI's API key as soon as you run the LLM-based code in the terminal.

