# The-application-of-facebook-prophet-on-time-series-predictions
A Kaggle project on the application of facebook prophet on time series predictions. The crime rate in chicago is the case study
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet

chicago_df_1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', error_bad_lines= False) # ignore all the bad lines
chicago_df_2 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', error_bad_lines= False)
chicago_df_3 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', error_bad_lines= False)

print(chicago_df_1.shape)
print(chicago_df_2.shape)
print(chicago_df_3.shape)

# Exploring the dataset
chicago_df = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3], ignore_index=False,axis=0)
print(chicago_df.head())

print(chicago_df.tail())

# To check how many elements that are missing on the dataset
plt.figure(figsize=(10,10))
sns.heatmap(chicago_df.isnull(), cbar=False, cmap='YlGnBu')
plt.show()

# we are interested in time series so we are going to drop all the columns we don't need
chicago_df.drop(['Unnamed: 0', 'Case Number', 'ID', 'IUCR', 'X Coordinate', 'Y Coordinate', 'Updated On', 'Year', 'FBI Code', 'Beat', 'Ward', 'Community Area', 'Location', 'District', 'Latitude', 'Longitude'], inplace= True, axis=1)
print(chicago_df.head())

chicago_df.Date = pd.to_datetime(chicago_df.Date, format = '%m/%d/%Y %I:%M:%S %p')
print(chicago_df.Date)

# To index the dataframe as having datetime as it's index
chicago_df.index = pd.DatetimeIndex(chicago_df.Date)


'''
To visualize the data, to know the number of cases where robbery, prostitution etc. 
the information is in primary column, therefor we use the value count function to achieve this'''
print(chicago_df['Primary Type'].value_counts())

# To locate the first or top 15 samples from the primary type

print(chicago_df['Primary Type'].value_counts().iloc[:15])

# to locate the index where the top 15 samples from the primary type belongs
order_data= chicago_df['Primary Type'].value_counts().iloc[:15].index

# to visualize the top 15 on seaborne
plt.figure(figsize=(15,10))
sns.countplot(y= 'Primary Type', data = chicago_df, order= order_data)
plt.show()

# to visualize the Location description on seaborne plot
plt.figure(figsize=(15,10))
sns.countplot(y= 'Location Description', data=chicago_df, order=chicago_df['Location Description'].value_counts().iloc[:15].index)
plt.show()

# to resample the dataset based on the years that is to check the frequency of occurrence of crimes per year
print(chicago_df.resample('Y').size())
# to visualize the above on a graph
plt.plot(chicago_df.resample('Y').size())
plt.title('Crimes Count Per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
plt.show()

#  to resample the dataset based on the months that is to check the frequency of occurrence of crimes per month
print(chicago_df.resample('M').size())

# to visualize the the above on a graph
plt.plot(chicago_df.resample('M').size())
plt.title('Crimes Count Per Month')
plt.xlabel('Months')
plt.ylabel('Number of Crimes')
plt.show()

# to resample the dataset based quarterly occurrences that is to check the frequency of occurrence of crimes per quarter of the year
print(chicago_df.resample('Q').size())

# to visualize the the above on a graph
plt.plot(chicago_df.resample('Q').size())
plt.title('Crimes Count Per Quarter of the year')
plt.xlabel('Quarter')
plt.ylabel('Number of Crimes')
plt.show()

# Preparation of the dataset in other to apply the facebook prophet per month of occurrences
chicago_prophet_df = chicago_df.resample('M').size().reset_index()
print(chicago_prophet_df)

# to rename the columns as Date and Crime Count
chicago_prophet_df.columns = ['Date', 'Crime Count']

# renaming the date column to ds and crime count column to Y in other to perform predictions.
chicago_prophet_df_final = chicago_prophet_df.rename(columns = {'Date':'ds', 'Crime Count': 'y'})

# to make predictions
m = Prophet()
m.fit(chicago_prophet_df_final)

future = m.make_future_dataframe(periods=365) # to forecast the future ie Simulate the trend using the extrapolated generative model.
forecast = m.predict(future) #Predict the future using the prophet model

# to show the crime rate predictions
figure = m.plot(forecast, xlabel= 'Date', ylabel= 'Crime Rate')
plt.show()

# to predict the seasonality occurrences with the year
figure = m.plot_components(forecast)
plt.show()


# To make preddictions of crimes in the next 2 years

future = m.make_future_dataframe(periods=730) # to forecast the future ie Simulate the trend using the extrapolated generative model.
forecast = m.predict(future) #Predict the future using the prophet model

# to show the crime rate predictions
figure = m.plot(forecast, xlabel= 'Date', ylabel= 'Crime Rate')
plt.show()

# to predict the seasonality occurrences with the year
figure = m.plot_components(forecast)
plt.show()
