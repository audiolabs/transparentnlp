# Transparent NLP

This repository contains code associated with the paper "Transparent NLP: Using RAG and LLM Alignment for Privacy Q&A" by Popp et al. (2025). Each sub-directory contains data and 
scripts to get the analysis results.

The step to run each sub.directory is as follows:

1. data_processing 
2. general_statistical_llm_based_metrics
3. percentage_comparison
4. PCA

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
    python3 -m venv venv
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

