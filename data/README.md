##Overview of the Data Directory

This directory houses multiple subdirectories, each containing datasets from 
various RAG and RAIN experiments.

It includes the original PrivacyQA dataset along with the designed answers 
evaluated by legal experts and designers, as well as non-proactive answers. 
The PrivacyQA dataset is stored in TSV format to prevent issues with commas 
in sentences affecting column separation.

Other subdirectories contain .json files or TSV files coming from each experiment, 
with the subdirectory names indicating the evaluation metrics used in 
those experiments.

For future experiments, please create a subdirectory named according 
to the metrics used and place the corresponding JSON files inside. 
This will help ensure smooth functioning of the data processing code.

