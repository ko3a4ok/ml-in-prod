# Pandas format benchmark

## Setup
Randomly generate 1,000,000 records. Each record consists of: 14-char string, integer number and short string. 

## Write results
|Format| Time to Save, seconds |
|------|--------------|
| csv | 2.19 |
|json| 0.75|
|parquet|0.38|
|xml|15.95|

## Read results
|Format| Time to Load, seconds |
|------|--------------|
| csv | 0.80|
|json|3.70|
|parquet|0.42|
|xml|31.04|

## Conclusion
Parquet data format works the best for loading and saving data. This data format is the best option to use for the project.
