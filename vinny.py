def create_prompt_template():
    prompt_template = """
    Analyze the SQL explain statement provided below, considering the given context and conditions:

    **Table Information:**
    - Rows: {rows}
    - Schema:
      {context_conditions}

    **Query Details:**
    I am running the following query with its explain statement:

    **SQL Explain Statement:**
    {explain_statement} 

    **Analysis Result:**
    Provide the output in JSON format with the following structure:
    {
        "Result": "Accepted or Rejected",
        "Reason": "Detailed reason to accept or reject. If rejected, include all possible reasons such as query_cost, filtering, use of joins, etc.",
        "Suggestions": "Suggestions on how to improve the query, if applicable."
    }
    """
    return prompt_template
