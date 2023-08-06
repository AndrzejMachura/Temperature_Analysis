# import libraries
import pandas as pd
from urllib.request import urlopen
import json
import ssl


def get_data(lat, lon, sd, ed, cd, tz):
    # dowload mean daily temperature, sum of daily snowfall and rain  
    # Weather data by Open-Meteo.com (https://open-meteo.com/)

    # store the URL in url as parameter for urlopen
    adres = "https://archive-api.open-meteo.com"
    url = adres+"/v1/archive?latitude="+str(lat)+"&longitude="+str(lon)+"&start_date="+sd+"&end_date="+ed+"&daily="+cd+"&timezone="+tz

    # store the response of URL
    context = ssl._create_unverified_context()
    response = urlopen(url, context=context)

    # storing the JSON response 
    # from url in data
    data_json = json.loads(response.read())

    # create pandas data frame from JSON String
    df = pd.DataFrame.from_dict(data_json["daily"], orient="columns")
    df["time"] = pd.to_datetime(df["time"])

    return df

if __name__ == "__main__":

    # localisation parameters
    # Warsaw latitude and longnitutde
    lat = 52.23
    lon = 21.01

    # date range for analysis
    sd = "1965-01-01"
    ed = "2023-01-31"

    # requested data
    cd = "temperature_2m_mean,rain_sum,snowfall_sum"

    # time zone
    tz = "Europe%2FBerlin"

    # run get data fuction and save data to csv file
    df = get_data(lat, lon, sd, ed, cd, tz)
    df.to_csv("cimate_data.txt")