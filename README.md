# Offline Evaluation for PrivacyQA Systems

### <font color="red">Status: In Progress - This project is not yet complete.</font>

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

3. **Upload Data**:
Upload your CSV or JSON files into the data directory. The data files should contain the outputs of the systems you wish to evaluate.



**Notes**:
1. You will be prompted to enter you OpenAI's API key as soon as you run the above code in the terminal.

## Contact
For any questions or issues, please contact Zahra Kolagar at <a href="mailto:zahra.kolagar@iis.fraunhofer.de">zahra.kolagar@iis.fraunhofer.de</a>
.