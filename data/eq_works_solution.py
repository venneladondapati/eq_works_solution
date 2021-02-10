import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
import geopy.distance
import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import math

def read_data():
      dataSample_df = spark.read.options(header=True,inferSchema='True').csv("DataSample.csv")
      POI_df = spark.read.options(header=True,inferSchema='True').csv("POIList.csv")
      return dataSample_df, POI_df

def cleanup(dataSample_df):
      dataSample_df = dataSample_df.dropDuplicates([dataSample_df.columns[1],dataSample_df.columns[5],dataSample_df.columns[6]])
      return dataSample_df

def compute_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    distance = geopy.distance.distance(coords_1, coords_2).km
    return distance

def assign_POI(lat1, lon1, POI_collect):
    min_distance = -1
    min_POIID = ""
    for row in POI_collect:
        row_dict = row.asDict()
        lat2 = float((row_dict[' Latitude']))
        lon2 = float((row_dict['Longitude']))
        POI_ID = row_dict['POIID']
        distance = compute_distance(lat1,lon1,lat2,lon2)
        if min_distance == -1:
            min_POIID = POI_ID
            min_distance = distance
        elif distance < min_distance:
            min_POIID = POI_ID
            min_distance = distance
    return min_POIID,min_distance

def rowwise_function(row, POI_collect):
    row_dict = row.asDict()
    lat1 = float((row_dict['Latitude']))
    lon1 = float((row_dict['Longitude']))
    row_dict['POI_label'],row_dict['POI_distance'] = assign_POI(lat1, lon1, POI_collect)
    newrow = Row(**row_dict)
    return newrow

def label_data(dataSample_df, POI_df):
      POI_collect = POI_df.collect()
      data_rdd = dataSample_df.rdd
      data_rdd_new = data_rdd.map(lambda row: rowwise_function(row, POI_collect))
      label_df = spark.createDataFrame(data_rdd_new)
      return label_df

def data_visualize(label_df):
      colorsIdx = {'POI1': 'red', 'POI3': 'green', 'POI4':'blue'}
      data = go.Scattergeo(
              lon = [float(row.Longitude) for row in label_df.collect()],
              lat = [float(row.Latitude) for row in label_df.collect()],
              mode = 'markers',
              marker=dict(size=2, color=[colorsIdx[row.POI_label] for row in label_df.collect()])
              )

      layout = dict(title = 'POI distibution',
                    geo_scope = 'north america'
                   )
      choromap = go.Figure(data = [data],layout = layout)
      choromap.show()
      return

def data_analysis(label_df):
      avg_df = label_df.groupby('POI_label').agg({"POI_distance":"avg"}).withColumnRenamed('avg(POI_distance)','avg')
      stddev_df = label_df.groupby('POI_label').agg({"POI_distance":"stddev"}).withColumnRenamed('stddev(POI_distance)','stddev')
      avg_df = avg_df.join(stddev_df, on=['POI_label'],how='inner')
      radius_df = label_df.groupby('POI_label').agg({"POI_distance":"max"}).withColumnRenamed('max(POI_distance)','radius')
      count_df = label_df.groupby('POI_label').agg({"POI_distance":"count"}).withColumnRenamed('count(POI_distance)','Count')
      density_df = radius_df.join(count_df,on=['POI_label'],how='inner')
      density_df = density_df.join(avg_df,on=['POI_label'],how='inner')
      def calculate_density(row):
          row_dict = row.asDict()
          radius = float((row_dict['radius']))
          count = float((row_dict['Count']))
          row_dict['density'] = count/math.pi*(radius**2)
          newrow = Row(**row_dict)
          # return new row
          return newrow
      density_rdd = density_df.rdd
      density_rdd_new = density_rdd.map(lambda row: calculate_density(row))
      analysis_df = spark.createDataFrame(density_rdd_new)
      return analysis_df

def popularity(row, total_count):
    row_dict = row.asDict()
    avg = float((row_dict['avg']))
    std_dev = float((row_dict['stddev']))
    count = float((row_dict['Count']))
    x = (avg/std_dev)+(count/total_count)
    sig_val = 1/(1 + math.exp(-x))
    normal_val = (sig_val-1)/2
    popularity = 10 * normal_val
    row_dict['popularity'] = popularity
    newrow = Row(**row_dict)
    return newrow

def assign_popularity(analysis_df, dataSample_df):
      total_count = dataSample_df.count()
      analysis_rdd = analysis_df.rdd
      analysis_rdd_new = analysis_rdd.map(lambda row: popularity(row, total_count))
      popularity_df = spark.createDataFrame(analysis_rdd_new)
      return popularity_df


if __name__ == "__main__":
    spark = SparkSession.builder.appName("eq_works_solution").config("spark.master", "local").getOrCreate()

    dataSample, poi = read_data()

    # Problem 1
    dataSample = cleanup(dataSample)
    dataSample.show()

    # Problem 2
    label = label_data(dataSample, poi)
    label.show()

    #data_visualize(label)

    # Problem 3
    analysis = data_analysis(label)
    analysis.show()

    # Problem 4
    popularity = assign_popularity(analysis, dataSample)
    popularity.show()

    spark.stop()