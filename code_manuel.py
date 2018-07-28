###############################################################################################################################
# Distribution of trip distances
# define the figure with 2 subplots
###FIGURE - 7 #####

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_2016_1 = pd.read_csv("s3://ludditiesnyctaxi/new/2016/yellow/yellow_2016_11.csv")
#%%
df_2016_1.info()
#%%
fig,ax = plt.subplots(1,2,figsize = (15,4)) 

# histogram of the number of trip distance
df_2016_1.Trip_distance.hist(bins=20,ax=ax[0], edgecolor='black')
ax[0].set_xlabel('Trip Distance (miles)')
ax[0].set_ylabel('Count')
ax[0].set_yscale('log')
ax[0].set_title('Histogram of Trip Distance')
# create a vector to contain Trip Distance
v = df_2016_1.Trip_distance 
# exclude any data point located further than 3 standard deviations of the median point  
v[~((v-v.median()).abs()>3*v.std())].hist(bins=20,ax=ax[1], edgecolor='black') 
ax[1].set_xlabel('Trip Distance (miles)')
ax[1].set_ylabel('Count')
ax[1].set_title('A. Histogram of Trip Distance (without outliers)')


plt.show()
#%%



###############################################################################################################################
###FIGURE - 3 #####
import pandas as pd
df_2016_1 = pd.read_csv("s3://ludditiesnyctaxi/new/2016/yellow/yellow_2016_11.csv")
#%%
df_2016_1.info()
#%%
df_2016_1.duplicated()
#%%
df_2016_1.shape
#%%
#Extract DateTime features
df_2016_1['pickup_datetime'] = pd.to_datetime(df_2016_1.lpep_pickup_datetime)
df_2016_1.loc[:, 'pickup_date'] = df_2016_1['pickup_datetime'].dt.date
df_2016_1.loc[:, 'pickup_weekday'] = df_2016_1['pickup_datetime'].dt.weekday
df_2016_1.loc[:, 'pickup_hour_weekofyear'] = df_2016_1['pickup_datetime'].dt.weekofyear
df_2016_1.loc[:, 'pickup_hour'] = df_2016_1['pickup_datetime'].dt.hour
df_2016_1.loc[:, 'pickup_minute'] = df_2016_1['pickup_datetime'].dt.minute
df_2016_1.loc[:, 'pickup_dt'] = (df_2016_1['pickup_datetime'] - df_2016_1['pickup_datetime'].min()).dt.total_seconds()
#%%
import matplotlib.pyplot as plt

# Distribution of trip distance by pickup hour
# Q: does time of the day affect the taxi ridership?


fix, axis = plt.subplots(1,1,figsize=(12,7))
#aggregate trip_distance by hour for plotting
tab = df_2016_1.pivot_table(index='pickup_hour', values='Trip_distance', aggfunc=('mean','median')).reset_index()
     
tab.columns = ['Hour','Mean_distance','Median_distance']
tab[['Mean_distance','Median_distance']].plot(ax=axis)
plt.ylabel('Metric (miles)')
plt.xlabel('Hours after midnight')
plt.title('Distribution of trip distance by pickup hour')
plt.xlim([0,23])
plt.show()



###############################################################################################################################
###FIGURE - 4 #####
#%%
import matplotlib.pyplot as plt
import seaborn as s
# DO PEOPLE LIKE TO TRAVEL IN GROUP/SHARE RIDES OR PREFER TO RIDE ALONE
# Q: Are people more likely to travel in groups during holiday seasons/
df_2016_1["pickup_month"] = pd.DatetimeIndex(df_2016_1['pickup_date']).month
jan_rides  = df_2016_1[(df_2016_1["pickup_month"] == 1)]
feb_rides  = df_2016_1[(df_2016_1["pickup_month"] == 2)]
mar_rides  = df_2016_1[(df_2016_1["pickup_month"] == 3)]
apr_rides  = df_2016_1[(df_2016_1["pickup_month"] == 4)]
may_rides  = df_2016_1[(df_2016_1["pickup_month"] == 5)]
jun_rides  = df_2016_1[(df_2016_1["pickup_month"] == 6)]
jul_rides  = df_2016_1[(df_2016_1["pickup_month"] == 7)]
aug_rides  = df_2016_1[(df_2016_1["pickup_month"] == 8)]
sep_rides  = df_2016_1[(df_2016_1["pickup_month"] == 9)]
oct_rides  = df_2016_1[(df_2016_1["pickup_month"] == 10)]
nov_rides  = df_2016_1[(df_2016_1["pickup_month"] == 11)]
dec_rides  = df_2016_1[(df_2016_1["pickup_month"] == 12)]

fig,ax = plt.subplots(1,1,figsize = (12,5))
s.countplot(x="Passenger_count",data=df_2016_1)
plt.title('Trips with number of passengers')




###############################################################################################################################
###FIGURE - 5 #####
df_2016_2 = pd.read_csv("s3://ludditiesnyctaxi/new/2016/yellow/yellow_tripdata_2016-01.csv")

import folium # goelogical map
map_1 = folium.Map(location=[40.767937,-73.982155 ],tiles='OpenStreetMap',
 zoom_start=12)
#tile: 'OpenStreetMap','Stamen Terrain','Mapbox Bright','Mapbox Control room'
for each in df_2016_2[:1000].iterrows():
    folium.CircleMarker([each[1]['pickup_latitude'],each[1]['pickup_longitude']],
                        radius=3,
                        color='red',
                        popup=str(each[1]['pickup_latitude'])+','+str(each[1]['pickup_longitude']),
                        fill_color='#FD8A6C'
                        ).add_to(map_1)
#change the directory
map_1.save("nyc_map.html")
###############################################################################################################################
FIGURE -8 MISSING RATIO CHECK
%pyspark
import pandas as pd
#2009 tamam
#2010 tamam
#2011 tamam
#2012 tamam

nyc_green_20161=pd.read_csv('s3://ludditiesnyctaxi/new/2012/yellow_2012_10_3.csv')
all_data_na = (nyc_green_20161.isnull().sum() / len(nyc_green_20161)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


###############################################################################################################################

FIGURE -6
3- Sort lines (taxi trips) wrt total_amount (charged to customers) in descending order and print first 15 of them…
>>> lines = sc.textFile("file:/home/ubuntu/taxidata/yellow_tripdata_2017-01.csv")
>>> content = lines.filter(lambda line: line != header)
>>> max_15_total_amount = content.map(lambda line: (line.split(",")[0], float(line.split(",")[4]), Output: line.split(",")[9], float(line.split(",")[16]))).takeOrdered(15, lambda x : -x[3])
>>> max_15_total_amount
Output: [('1', 0.0, '2', 625901.6), ('1', 0.0, '3', 538580.0), ('2', 0.0, '2', 9001.3), ('1', 15.9, '4', 8043.84), ('1', 6.0, '3', 3009.8), ('1', 1.9, '2', 3009.3), ('1', 0.5, '1', 2000.28), ('1', 4.1, '1', 2000.28), ('1', 0.0, '1', 2000.28), ('1', 0.7, '1', 1000.29), ('1', 17.0, '2', 963.88), ('1', 7.8, '4', 930.34), ('1', 0.8, '3', 899.34), ('1', 5.0, '3', 824.38), ('1', 17.0, '3', 776.3)]

###############################################################################################################################


4- Calculate sum of total_amount variable wrt vendors_id…
>>> vendors_total_revenue = content_2.map(lambda line: (line.split(",")[0], 
Output: float(line.split(",")[16]))).reduceByKey(lambda x,y: x+y)
>>> vendors_total_revenue.collect()
Output: [('1', 67916970.82997812), ('2', 82849477.31987293)]

5- Calculate average total_amount wrt vendors…

Firstly, we should add a ‘counter’ to each line:
>>> vendors_average_revenue_1 = content_2.map(lambda line: (line.split(",")[0], 
Output: float(line.split(",")[16]))).mapValues(lambda x: (x,1))
>>> vendors_average_revenue_1.take(10)
Output: [('1', (15.3, 1)), ('1', (7.25, 1)), ('1', (7.3, 1)), ('1', (8.5, 1)), ('2', (52.8, 1)), ('1', (5.3, 1)), ('2', (27.96, 1)), ('1', (8.75, 1)), ('1', (8.3, 1)), ('2', (8.3, 1))]


Then we should sum each total_amount and counter wrt vendor_id:
>>> vendors_average_revenue_2 = content_2.map(lambda line: (line.split(",")[0], 
Output: float(line.split(",")[16]))).mapValues(lambda x: (x,1)).reduceByKey(lambda x,y : (x[0] + y[0], x[1] + y[1]))
>>> vendors_average_revenue_2.collect()
Output: [('1', (67916970.82997812, 4397921)), ('2', (82849477.31987293, 5312203))]

Finally, we have to dive the total total_amount by total count for each vendor:
>>> vendors_average_revenue_3 = vendors_average_revenue_2.mapValues(lambda x: x[0]/x[1])
>>> vendors_average_revenue_3.collect()
Output: [('1', 15.442971992898036), ('2', 15.596067642722414)]



###############################################################################################################################
#Reading data as spark dataframe... 
%pyspark
df_2017_1 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_01.csv")
df_2017_2 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_02.csv")
df_2017_3 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_03.csv")
df_2017_4 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_04.csv")
df_2017_5 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_05.csv")
df_2017_6 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_06.csv")
df_2017_7 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_07.csv")
df_2017_8 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_08.csv")
df_2017_9 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_09.csv")
df_2017_10 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_10.csv")
df_2017_11 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_11.csv")
df_2017_12 = spark.read.format("csv").option("header", "true").load("s3://ludditiesnyctaxi/new/2017/yellow/yellow_2017_12.csv")

#Concat the data to make one year file
%pyspark
from functools import reduce
from pyspark.sql import DataFrame

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

datamerge = unionAll(df_2017_1,df_2017_2,df_2017_3,df_2017_4,df_2017_5,df_2017_6,df_2017_7,df_2017_8,df_2017_9,df_2017_10,df_2017_11,df_2017_12)


#Register the data as table 
%pyspark
datamerge=datamerge.cache()
datamerge.createOrReplaceTempView("datamerge")

#figure -  9
%sql
SELECT passenger_count, AVG(fare_amount)
FROM datamerge 
WHERE passenger_count > 0 and passenger_count < 7 
GROUP BY passenger_count 


#figure -  10
%sql
SELECT passenger_count, COUNT(fare_amount) as fare_amount
FROM datamerge 
WHERE passenger_count > 0 and passenger_count < 7 
GROUP BY passenger_count 


%sql
SELECT ROUND(PULocationID, 4) AS lat,
ROUND(DOLocationID, 4) AS long,
COUNT(*) AS num_pickups,
SUM(fare_amount) AS total_revenue,
SUM(fare_amount)/COUNT(*) AS avr_revenue
FROM datamerge
WHERE fare_amount/trip_distance BETWEEN 2 AND 10
GROUP BY lat, long



