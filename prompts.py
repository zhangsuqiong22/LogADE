# A-series: No RAG context

PROMPT_A1 = """Your task is to determine whether the input log template is normal or anomaly.
Input Log Template: {query}

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly.
You should generate reasons for your judgment.

Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""

PROMPT_A2 = """Your task is to determine whether the input log template is normal or anomaly.
Input Log Template: {query}

Analyze the Input Log Template independently:
 - You need to carefully check the text content for keywords. Identify key elements such as error codes, status messages, and other significant terms.
 - In the log template, the parameters are replaced by <*>. You should never consider <*> or missing values as the reason for abnormality.

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly.
Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""

# B-series: RAG not hit, fallback to independent analysis

PROMPT_B = """Your task is to determine whether the input log template is normal or anomaly.

Input Log Template: {query}

No similar anomaly cases were found in the knowledge base.
Please analyze the log independently:
 - Identify key elements such as error codes, status messages, and other significant keywords.
 - Do not consider <*> placeholders or missing values as evidence for anomalies.

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly.
Output format: Return in JSON format with keys: is_anomaly, reason
Answer:"""

# C1-1: One anomaly case hit in RAG
PROMPT_C1_1 = """Your task is to determine whether the input log is anomalous based on a single retrieved anomaly case.

Input Log Template:
{query}

Actual log message:
{log_content}

Retrieved Anomalous Log Case with Explanation:
{SimilarAnomaly}

ANALYSIS INSTRUCTIONS:
Since we retrieved exactly one anomaly case, carefully analyze whether the current log matches the retrieved case.

Steps to follow:
1. First, confirm the log as anomalous (is_anomaly = 1).
2. Extract the explanation from the retrieved anomaly case.
3. Analyze how well the current log matches the retrieved case.
4. Extract any relevant contextual information from the log message.
5. Identify any case ID mentioned in the retrieved case.

Output format (in JSON):
{{
  "is_anomaly": 1,
  "reason": "Explanation from the retrieved anomaly case",
  "matched_case_id": "Relevant case ID if available",
  "confidence": "High/Medium/Low based on similarity",
  "extracted_context": {{
    "relevant fields extracted from the actual log message"
  }}
}}

Remember: Since only one anomaly case was retrieved, focus on providing a clear explanation of why 
this log matches the anomaly pattern in the retrieved case.
Answer:"""

# C1-2: Multiple anomaly cases hit, with consistent explanations to be merged

PROMPT_C1_2 = """Your task is to determine whether the input log is anomalous based on retrieved anomaly cases.

Input Log Template:
{query}

Actual log message:
{log_content}

Retrieved Anomalous Log Cases with Explanations:
{SimilarAnomaly}

ANALYSIS INSTRUCTIONS:
Since we retrieved multiple anomaly cases with consistent explanations, this log should be classified as anomalous.
Your task is to merge the explanations into a single, comprehensive reason.

Steps to follow:
1. First, confirm the log as anomalous (is_anomaly = 1).
2. Identify common themes across all retrieved explanations.
3. Merge the explanations, removing redundancies while preserving all unique insights.
4. Extract any relevant contextual information from the log message.
5. Identify any case IDs mentioned in the retrieved cases.

Output format (in JSON):
{{
  "is_anomaly": 1,
  "reason": "Comprehensive merged explanation from all retrieved cases",
  "matched_case_ids": ["Relevant case IDs if available"],
  "extracted_context": {{
    "relevant fields extracted from the actual log message"
  }}
}}

Remember: Since the retrieved cases are consistent, focus on creating a unified, thorough explanation 
that captures all important aspects from the individual explanations.
Answer:"""

# C2-1: Multiple anomaly cases hit, with conflicting or diverse explanations
PROMPT_C2_1 = """Your task is to determine whether the input log is anomalous based on multiple retrieved anomaly cases with potentially conflicting explanations.

Input Log Template:
{query}

Actual log message:
{log_content}

Retrieved Anomalous Log Cases with Explanations:
{SimilarAnomaly}

ANALYSIS INSTRUCTIONS:
You have retrieved multiple anomaly cases with diverse or conflicting explanations. You need to carefully analyze whether the current log is truly anomalous.

Steps to follow:
1. Analyze each retrieved case and its explanation individually.
2. Compare the input log with each retrieved case to assess similarity.
3. Determine if the input log aligns more closely with a specific type of anomaly.
4. Weigh the evidence for and against classifying this log as anomalous.
5. Make a final determination based on the strength of evidence.
6. Extract any relevant contextual information from the log message.
7. Identify case IDs mentioned in the retrieved cases that are most relevant.

Output format (in JSON):
{
  "is_anomaly": 0 or 1,
  "reason": "Detailed explanation of your decision, including which cases were most influential",
  "matched_case_ids": ["Relevant case IDs if available"],
  "confidence": "High/Medium/Low based on the strength of evidence",
  "extracted_context": {
    "relevant fields extracted from the actual log message"
  }
}

Remember: Unlike cases where explanations are consistent, here you must critically evaluate conflicting information and make a reasoned judgment about whether this log is truly anomalous.
Answer:"""
