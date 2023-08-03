import time
import datetime
import pytz
import pandas as pd

if __name__=="__main__":
    pred = Predict()
    db=DB()
    while True:
        now = datetime.datetime.now()
        seconds_until_next_minute = 60 - now.second - now.microsecond / 1E6
        time.sleep(seconds_until_next_minute)

        result_dict={}
        df=db.fetch_data_from_db()
        temp1=pred.inf_prepare_dataset(df,2)
        df = pd.DataFrame(temp1)     
        df.drop(15,inplace=True,axis=1)
        tz = pytz.timezone('UCT')
        now = datetime.datetime.now(tz)
        one_min_later = now + datetime.timedelta(minutes=1)
        time_str = one_min_later.strftime("%m-%d-%Y, %H:%M:%S %p")
        print(f"--------------- CPU Prediction for {time_str} ---------------")
        result_dict['Time']=one_min_later
        traces_pred,cpu_pred=predict(df)
        result_dict['Predicted CPU']=cpu_pred
        result_dict['Traces Result']=str(traces_pred)
        df = pd.DataFrame(result_dict, index=[0]) 
        push_result_to_DB(df)
