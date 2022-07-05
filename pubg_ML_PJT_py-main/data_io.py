import pandas as pd


base_path = "/Users/krc/Downloads/pubg-finish-placement-prediction/"


def load_dataset(csv_name):
   try:
     df = pd.read_csv(base_path + csv_name)
   except: 
     raise Exception("csv파일의 위치를 입력하세요!")
   return df


