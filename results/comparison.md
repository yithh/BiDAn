# Perbandingan tanpa Spark dan dengan Spark MLlib

## Akurasi
| percobaan   | model                            |   MAE_sklearn |   RMSE_sklearn |   R2_sklearn |   MAE_spark |   RMSE_spark |   R2_spark |
|:------------|:---------------------------------|--------------:|---------------:|-------------:|------------:|-------------:|-----------:|
| k60         | Dasar: jam sama minggu lalu      |        16.049 |         30.177 |        0.9   |      16.049 |       30.177 |      0.9   |
| k60         | Dasar: rata-rata 24 jam terakhir |        47.242 |         74.747 |        0.388 |      47.242 |       74.747 |      0.388 |
| k60         | Ridge                            |        14.359 |         25.533 |        0.929 |      14.352 |       25.528 |      0.929 |
| k60         | GBT                              |        12.2   |         21.579 |        0.949 |      12.689 |       23.089 |      0.942 |
| k60         | GBT + cuaca                      |        13.129 |         23.576 |        0.939 |      14.356 |       25.847 |      0.926 |

## Waktu (detik)
| percobaan   | model             |   baris_latih |   zona |   fit_sklearn_s |   fit_spark_s |
|:------------|:------------------|--------------:|-------:|----------------:|--------------:|
| k60         | Ridge             |         24480 |     60 |            0.03 |          4.31 |
| k60         | GBT               |         24480 |     60 |            1.35 |        510.72 |
| k60         | GBT + cuaca       |         24480 |     60 |            1.26 |        528.94 |
| k60         | (pembuatan fitur) |         24480 |     60 |            0.21 |         64.62 |