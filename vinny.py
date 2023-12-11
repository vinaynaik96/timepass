import os
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

def read_excel_from_directory(file_path):
    if os.path.exists(file_path):
        # Read the Excel file using pandas
        df = pd.read_excel(file_path)
        return df
    else:
        return None

@app.get("/read_excel/")
async def read_excel():
    directory = "your_directory_path_here"  # Specify your directory path here
    file_name = "your_excel_file.xlsx"  # Specify your Excel file name
    
    excel_file_path = os.path.join(directory, file_name)
    data = read_excel_from_directory(excel_file_path)
    
    if data:
        # Get column values
        columns = data.columns.tolist()
        return {"columns": columns, "data": data.to_dict(orient='records')}
    else:
        return {"error": "File not found or unable to read."}
