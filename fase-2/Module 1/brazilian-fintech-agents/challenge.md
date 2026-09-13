# Brazilian Fintech Agents

## Proposal

### Context

One Brazilian Fintech needs to automate your monthly analsys transaction database. Nowadays, analysts spends days consolidating  datas from multiply sources, identifing anomalies and generate executive reports. Your team was hired to develop an capable agent to perform their work autonomously.   

### Goal

Design a agent flow that receive an csv file o financial transaction. Execute a complete analysis pipeline and produce an MD executive report, with finding, alerts and recommendations.

#### Dataset 
 ./dataset/creditcard.csv


### Required Architecture:
The system must be an compose agent with 3 responsabilities;
1. Ingestion: validate the csv structure and rows, checking null values and data types, treating null values, converting types and response a basic static summary (shape, dtypes, missing values, date range)
2. EDA (Exploratory analysis): Calculate distribution per category and channel, identify the more actives customers, compute avarage ticket and standart deviation per customer segment, and detect temporal trends (week by week variation);  
3. Report Writter: Draft the executive report in markdown, including executive summary, analysis section, anomalies table and 3 actionable insights. 

### Technical requirements
The system must be exclusively implemented with data flow, using the pythin language.

#### Delivarables
1. Source code: Public Github repository (no secrets, no keys, no personal info)
2. Architecture diagram: Visual representation flow
3. Technical report: (1 or 2 pages) containing: description of architecture decisions, justifications for detection strategies of choosen anomalies. 

