
<h1><center>Unsupervised Anomaly Sensor Detection - IMS Bearing NASA Acoustics and Vibration Database </center></h1>

## PyCaret vs Custom BiLSTM (Bidirectional LSTM)

## Introduction


## Data
The data was source from [kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset) and comprises three vibrational sensor readings from the NASA Acoustics and Vibration Database. The datasets contained text files with 1-second vibration signal snapshots (20,480 data points) recorded at 5 and 10 minute intervals at a sampling rate of 20 kHz.

## Methodos

### Experimental setup PyCaret Models

|ID|Name|Reference|
|---|---|---|
|cluster|Clustering-Based Local Outlier|pyod\.models\.cblof\.CBLOF|
|iforest|Isolation Forest|pyod\.models\.iforest\.IForest|
|histogram|Histogram-based Outlier Detection|pyod\.models\.hbos\.HBOS|
|knn|K-Nearest Neighbors Detector|pyod\.models\.knn\.KNN|
|svm|One-class SVM detector|pyod\.models\.ocsvm\.OCSVM|
|mcd|Minimum Covariance Determinant|pyod\.models\.mcd\.MCD|

### Experimental Setup for BiLSTM

All experiments were run for 200 epochs, learning rate of 2 e-4 and batch size of 32. The architecture use was a configurable Bidirectional-LSTM. For this work only one layer of size 32 was used. The encoder-decoder can be costumized to multiple bilstm layers.

|Exp|Model|Loss|Optim|
|---|---|---|---|
|1|bilstm|mae\_loss|adam|
|2|bilstm|huber\_loss|adam|
|3|bilstm|mae\_loss|adamw|
|4|bilstm|huber\_loss|adamw|

## Results

### PyCaret



### BiLSTM - PyTorch


## References
[PyCaret](https://pycaret.gitbook.io/docs/)
