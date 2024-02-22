import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


prompt_template = """
    Analyze the following SQL explain statement, considering the provided context and conditions: 
    The table consists {rows} rows and the schema of the table is described below: 
    **Schema:** 
    {context_conditions} 
    I want to run a query whose explain statement I have generated and is described below as: 
    **SQL Explain Statement:** 
    {explain_statement} 
    
    **Based on your analysis, should the SQL query be accepted or rejected? Return the output strictly in json format in key-value type like Result: 
    Accepted or Rejected; Reason: Detailed reason to accept or reject - If rejected include all possible reasons to reject the query.
    e.g. query_cost, filtering, use of joins or anything which is applicable; Suggestions :how to improve the query; ** 
    """

data_input = {
    'explain_plan': {
        'query_block': {
            'select_id': 1,
            'cost_info': {
                'query_cost': '488989.65'
            },
            'table': {
                'table_name': 'Customers',
                'access_type': 'ALL',
                'rows_examined_per_scan': 4569768,
                'rows_produced_per_join': 4569768,
                'filtered': '100.00',
                'cost_info': {
                    'read_cost': '32012.85',
                    'eval_cost': '456976.80',
                    'prefix_cost': '488989.65',
                    'data_read_per_join': '8G'
                },
                'used_columns': ['customer_id', 'customer_name', 'email', 'phone_number', 'address']
            }
        }
    },
    'schema': {
        'customer_id': ('int', 'NO', 'PRI', None, ''),
        'customer_name': ('varchar (100)', 'YES', '', None, ''),
        'email': ('varchar(100)', 'YES', '', None, ''),
        'phone_number': ('varchar(20)', 'YES', '', None, ''),
        'address': ('varchar(255)', 'YES', '', None, '')
    },
    'row_count': 4999999,
    'sql_query': 'select * from Customers; '
}

explain_plan = data_input["explain_plan"]
schema = data_input["schema"]
row_count = data_input["row_count"]
sql_query = data_input["sql_query"]

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyDKLRSlIMDQnCwMNtEOsSSE5zp9cmFprpY")

# Create a prompt template for SQL analysis
sql_analysis_prompt = PromptTemplate.from_template(prompt_template)

# Create an LLMChain with the new SQL analysis prompt
sql_analysis_chain = LLMChain(llm=llm, prompt=sql_analysis_prompt)

if __name__ == "__main__":
    # Generate SQL analysis response
    sql_analysis_resp = sql_analysis_chain.run(rows=row_count, context_conditions=schema, explain_statement=explain_plan)

    # Print the JSON-formatted response
    print(json.dumps(sql_analysis_resp, indent=2))
