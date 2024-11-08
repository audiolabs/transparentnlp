## Overview of the Percentage Comparison Directory

This directory holds the code for getting the comparison between the expert generated responses to responses 
collected from VaniallaRAG, RAG+RAIN, and RAG+MultipleRAIN.

### merge_all_metric_scores.py

This scripts merges all the metrics's scores collected previously into one dataset.

### clean_all_metrics_scores.py

This script cleans all the metric's scores for further processing.

### Percentage_as_good_as_human.py

This code is refactored code written by Birgit to compare the percentage to which the answers are similar to
the expert answers. You do not need to run this code as it is imported in the visulaization code described
below.

### visulaize_percentage.py
this code visualizes the result from the Percentage_as_good_as_human.py script.