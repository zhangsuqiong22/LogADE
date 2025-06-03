# A-series: No RAG context

PROMPT_A1 = """Your task is to determine whether the input log template is normal or anomaly.
Input Log Template: {question}

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly.
You should generate reasons for your judgment.

Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""

PROMPT_A2 = """Your task is to determine whether the input log template is normal or anomaly.
Input Log Template: {question}

Analyze the Input Log Template independently:
 - You need to carefully check the text content for keywords. Identify key elements such as error codes, status messages, and other significant terms.
 - In the log template, the parameters are replaced by <*>. You should never consider <*> or missing values as the reason for abnormality.

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly.
Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""

# B-series: RAG not hit, fallback to independent analysis

PROMPT_B = """Your task is to determine whether the input log template is normal or anomaly.

Input Log Template: {question}

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
{question}

Retrieved One Similar Anomalous Log Case with Explanation:
{context}

If the input log is structurally or semantically similar to the retrieved case, classify it as anomaly and use the given explanation.

Output format (in JSON):
{
  "is_anomaly": 1,
  "reason": "{explanation from context}"
}
Answer:"""

# C1-2: Multiple anomaly cases hit with similar meaning

PROMPT_C1_2 = """Your task is to determine whether the input log is anomalous.

Input Log Template:
{question}

Retrieved Multiple Anomalous Log Cases with Explanations:
{context}

Steps:
1. If the input log matches any retrieved case in structure or meaning, classify it as anomaly.
2. If multiple explanations have similar root causes (e.g., timeout, failure), select the most general or most matching one as the explanation.
3. If they conflict, defer to independent reasoning.

Output format (in JSON):
{
  "is_anomaly": 1,
  "reason": "..."
}
Answer:"""

PROMPT_C1_2 = """Your task is to determine whether the input log is anomalous and extract relevant context information (e.g., pod name, namespace) from the log message.

Input Log Template:
{question}

Retrieved Multiple Anomalous Log Cases with Explanations:
{context}

Steps:
1. If the input log matches any retrieved case in structure or meaning, classify it as anomaly.
2. If multiple explanations have similar root causes (e.g., timeout, failure), select the most general or most matching one as the explanation.
3. If they conflict, defer to independent reasoning.
4. Parse the input log message and extract key contextual details (such as pod name, namespace, container name, etc.).

Output format (in JSON):
{
  "is_anomaly": 1,
  "reason": "...",
  "context": "..."
}
Answer:"""


# C2-1: Multiple anomaly cases hit, with conflicting or diverse explanations

PROMPT_C2_1 = """Your task is to determine whether the input log is anomalous, based on multiple potentially conflicting anomaly explanations retrieved from a known case base.

Input Log Template:
{question}

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
