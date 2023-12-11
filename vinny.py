import os
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

def search_value_by_keyword(csv_path, keyword):
    if os.path.exists(csv_path):
        # Read the CSV file using pandas
        df = pd.read_csv(csv_path)
        
        # Search for the keyword in the 'Keyword' column
        filtered_data = df[df['Keyword'] == keyword]
        
        if not filtered_data.empty:
            # Get the value corresponding to the keyword
            return filtered_data['Value'].iloc[0]
        else:
            return None
    else:
        return None

@app.get("/search/")
async def search_value(keyword: str):
    csv_file_path = "path_to_your_csv_file/data.csv"  # Specify your CSV file path here
    result = search_value_by_keyword(csv_file_path, keyword)
    
    if result is not None:
        return {"keyword": keyword, "value": result}
    else:
        return {"error": "Keyword not found or file not found."}
