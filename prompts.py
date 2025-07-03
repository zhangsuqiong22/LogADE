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

PROMPT_C1_1 = """Your task is to determine whether the input log is anomalous.

Input Log Template:
{query}

Retrieved One Similar Anomalous Log Case with Explanation:
{context}

If the input log is structurally or semantically similar to the retrieved case, classify it as anomaly and use the given explanation.

Output format (in JSON):
{
  "is_anomaly": 1,
  "reason": "{explanation from context}"
}
Answer:"""



PROMPT_C1_2 = """Your task is to determine whether the input log is anomalous based on retrieved anomaly cases.

Input Log Template:
{query}

Actual log message:
{log_content}

Retrieved Anomalous Log Cases with Explanations:
{SimilarAnomaly}

CRITICAL INSTRUCTION:
If {SimilarAnomaly} contains ANY text (is not empty), you MUST classify this log as an anomaly (is_anomaly = 1) WITHOUT ANY CONTENT ANALYSIS.
Do NOT analyze whether the log content semantically matches the retrieved cases.
The mere presence of retrieved cases is the ONLY criteria for anomaly classification.

Steps to follow STRICTLY:
1. Check if {SimilarAnomaly} is empty or not.
2. If {SimilarAnomaly} is NOT empty, set is_anomaly = 1 and use text from {SimilarAnomaly} as the reason.
3. If {SimilarAnomaly} is empty, set is_anomaly = 0.
4. Extract contextual information from the log message if needed.

Output format (in JSON):
{{
  "is_anomaly": 1 if {SimilarAnomaly} contains ANY text, otherwise 0,
  "reason": "Direct explanation from the retrieved anomalous cases",
  "matched_case_ids": ["Relevant case IDs if available"],
  "extracted_context": {{
    "relevant fields extracted from the actual log message"
  }}
}}

Remember: Do NOT perform semantic comparison between the input log and the retrieved cases.
The only decision criterion is whether {SimilarAnomaly} contains text or not.
Answer:"""
# C2-1: Multiple anomaly cases hit, with conflicting or diverse explanations

PROMPT_C2_1 = """Your task is to determine whether the input log is anomalous, based on multiple potentially conflicting anomaly explanations retrieved from a known case base.

Input Log Template:
{query}

Actural log message:
{log_content}

Retrieved Anomalous Log Cases with Explanations:
{context}

Please reason step by step:
1. Analyze each retrieved case and explanation.
2. Determine whether the input log is more similar to a specific class of anomaly.
3. If similarity is weak or conflicting, analyze the log independently.
4. Ignore placeholders like <*>.

Output format (in JSON):
{
  "is_anomaly": 0 or 1,
  "reason": "...",
  "matched_case_ids": ["..."]
}
Answer:"""
