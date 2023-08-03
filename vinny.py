def push_result_to_DB(df):
    password = "merck@1234"
    encoded_password = urllib.parse.quote(password, safe='')
    engine = create_engine(f"postgresql://merck_svc:{encoded_password}@localhost/merck_db")
    record_ins=df.to_sql('prediction_result1', con=engine, if_exists='append', index=False)
    print("records pushes to db is",record_ins)
    return record_ins  
