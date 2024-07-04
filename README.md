# Offline Evaluation for PrivacyQA Systems

### <span style="color:red">Status: In Progress - This project is not yet complete..</span>

##Overview
This repository contains code for offline evaluation of three different systems designed to handle PrivacyQA for virtual assistants. These systems process privacy-related questions over Alexa's privacy policies and documents and generate responses based on retrieved documents. The systems under evaluation are:

1. RAG System: A plain RAG (Retrieval-Augmented Generation) implementation by the NLU team, adapted to handle privacy policies and documents.

2. RAG + RAIN System: This system enhances the plain RAG system by incorporating RAIN, a guardrail that evaluates and corrects the output of the large language model at the token level based on the criterion of comprehensibility.

3. RAG + MultiRAIN System: This system further extends the RAG + RAIN system by adding another criterion, honesty, to evaluate and correct the output of the large language model at the token level.
Each system takes a user query as input, retrieves relevant documents, and generates a response to the user query.

## Offline Evaluation
The offline evaluation focuses on the final output of each system and compares them at two levels:

Between-System Evaluation and Comparison: This level compares the final output of each system against one another on semantic similarity to determine which systems produce semantically similar responses to the user query.

LLM Response Evaluation on Several Metrics: This level evaluates the final output of each system on multiple metrics. The metrics to be used for evaluation will be updated later.


## Running the Code

To run the code, follow these steps:

1. **Clone the Repository**: 
   ```bash
   git clone <repository_url>
   cd <repository_name>

2. **Install Dependencies**:
Ensure you have Poetry installed. Initialize the environment and install dependencies:
    ```bash
    poetry install

3. **Upload Data**:
Upload your CSV or JSON files into the data directory. The data files should contain the outputs of the systems you wish to evaluate.


4. **Run the Evaluation**:
Execute the evaluation.py script to perform the evaluation:
    ```bash
   poetry run python privacyQA_offline_evaluation/evaluation.py /path/to/input/file.csv output_file_name.csv

**Notes**:
1. You will be prompted to enter you OpenAI's API key as soon as you run the above code in the terminal.
2. Define the path to your input file and rename your output file in the code above.

## Directory Structure

    offline_evaluation/
    │
    ├── output_dir/
    │   └── # it will be created automatically
    ├── offline_evaluation/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── evaluation.py
    │   ├── utils.py
    │
    ├── .gitignore
    ├── pyproject.toml
    ├── requirements.txt
    └── README.md

## Contact
For any questions or issues, please contact Zahra Kolagar at [zahra.kolagar@iis.fraunhofer.de].