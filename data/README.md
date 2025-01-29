## Overview of the Data Directory

This directory houses multiple subdirectories, each containing datasets from 
various RAG and RAIN experiments.

The original PrivacyQA dataset along with the designed answers evaluated by legal experts and designers, 
as well as non-proactive answers has to be downloaded from the reporsitory as published together with the 
paper "Expert-Generated Privacy Q&A Dataset for Conversational AI and User Study Insights" 
by Anna Leschanowsky, Farnaz Salamatjoo, Zahra Kolagar and Birgit Popp (2025). The link to this repository is https://gitlab.cc-asp.fraunhofer.de/hmi/data-governance/privacy-qa-dataset. 
After downloading the PrivacyQA dataset, it should be placed in a folder within the directory transparentnlp/data 
for example as in transparentnlp/data/privacyQA.

Other subdirectories contain .json files or TSV files coming from each experiment, 
with the subdirectory names indicating the evaluation metrics used in 
those experiments.

For future experiments, please create a subdirectory named according 
to the metrics used and place the corresponding JSON files inside. 
This will help ensure smooth functioning of the data processing code.

