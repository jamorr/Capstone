# NYC - 311
This repository hosts files for analysis of New York City 311 data. The focus of this analysis is on the response time of the NYPD to 311 calls using survival analysis and machine learning methods. Additionally, analysis is performed on 311 volume for multiple NYC agencies using time series analysis methods like ARIMAX and VARIMAX as well as heirarchical reconciliation for better predictions for area subdivisions of New York (i.e. make sure that the sum of the volume of calls predicted for all boroughs equals the volume of calls predicted for the whole city).

Data was sourced from the NYC Open Data website, NOAA, and NCEI. Data was sourced from NYC open data using an app token, and users may experience throttling if they do not register for one (its free) on Socrata.

[311-Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data)  
[Crime Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data)   
[NYPD Precinct Boundaries](https://data.cityofnewyork.us/Public-Safety/Police-Precincts/78dh-3ptz)   
[NYPD Sector Boundaries](https://data.cityofnewyork.us/Public-Safety/NYPD-Sectors/eizi-ujye)   
[Hourly Weather Data](https://www.ncei.noaa.gov/access/search/data-search/global-hourly?dataTypes=TMP&dataTypes=DEW&dataTypes=AA1&dataTypes=VIS&dataTypes=WND&bbox=41.000,-74.261,40.558,-73.499&pageNum=1&startDate=2010-01-01T00:00:00&endDate=2023-02-01T23:59:59)  
[Major Storms Data](https://www.ncdc.noaa.gov/stormevents/listevents.jsp?eventType=ALL&beginDate_mm=01&beginDate_dd=01&beginDate_yyyy=2010&endDate_mm=11&endDate_dd=30&endDate_yyyy=2023&county=BRONX%3A5&county=KINGS%3A47&county=NEW%2BYORK%3A61&county=QUEENS%3A81&county=RICHMOND%3A85&hailfilter=0.00&tornfilter=0&windfilter=000&sort=DT&submitbutton=Search&statefips=36%2CNEW+YORK)  


## Directory Structure
├─/data  
│  ├─/311  
│  ├─/combined  
│  ├─/crime  
│  ├─/nypd  
│  └─/weather  
├─/models  
├─/plots  
│  └─animations  
├─/src  
├─README.MD  
└─requirements.txt  
