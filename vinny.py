
import pandas as pd
from sqlalchemy import create_engine
import urllib

def fetch_data_from_db():
    password = "merck@1234"
    encoded_password = urllib.parse.quote(password, safe='')
    engine = create_engine(f"postgresql://merck_svc:{encoded_password}@localhost/merck_db")
    
    query = "SELECT * FROM prediction_result1"
    df = pd.read_sql_query(query, engine)

    return df
