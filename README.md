
<h1 align="center">Anomaly Sensor Detection - IMS Bearing NASA Acoustics and Vibration Database</h1>

<h2 align="center">PyCaret vs Custom BiLSTM (Bidirectional LSTM)</h2>

## Introduction


## Data
The data was source from [kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset) and comprises three datsets of vibrational sensor readings from the NASA Acoustics and Vibration Database. The datasets contained text files with 1-second vibration signal snapshots (20,480 data points) recorded at 5 and 10 minute intervals at a sampling rate of 20 kHz.

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
#### Train dataset - Dataset 2 (avg_df2)

<p align="center">
  <img height="1200" src="plots/scatter_anomalies.png" width="600" alt="Anomaly distribution on train datset"/>
</p>
<p align="center">
Figure 2. 
</p>

#### Test dataset - Dataset 3 (avg_df3)

<p align="center">
  <img alt="Fig" height="500" src="plots/merged_test_anomaly_prediction.png" width="600"/> 
</p>

<p align="center">
  Figure 3. 
</p>
  
<br>
</br>


### BiLSTM - PyTorch

### Train dataset - Dataset 2 (avg_df2)

<p align="center">
<img src="plots/merged_bilstm_val_pred_anom_exp1.png" width="400" height="150" alt="train_anomaly exp1"/>A <img height="150" src="plots/merged_bilstm_val_pred_anom_exp2.png" title="train anomaly exp2" 
width="400"/>B
<img alt="train anomalies exp3" height="150" src="plots/merged_bilstm_val_pred_anom_exp3.png" width="400"/>C <img height="150" src="plots/merged_bilstm_val_pred_anom_exp4.png" width="400" title="train anomalies exp4"/>B
</p>

<p align="center">
  Figure 4. Anomalies distribution detected on the train dataset. The experimental setup is outlined in Table 2. A. Exp-01, B.Exp-02, C. Exp-03, D.Exp-04
</p>


<br>
</br>

### Test dataset - Dataset 3 (avg_df3)
<p align="center">
<img src="plots/merged_test_bilstm_anom_exp1.png" width="400" height="150" alt="test_anomaly exp1"/>A <img src="plots/merged_test_bilstm_anom_exp2.png" width="400" height="150" alt="test_anomaly exp2"/>B
<img src="plots/merged_test_bilstm_anom_exp3.png" width="400" height="150" alt="test_anomaly exp3"/>C <img src="plots/merged_test_bilstm_anom_exp4.png" width="400" height="150" alt="test_anomaly exp4"/>D
</p>
</p>
<p align="center">
Figure 5. Anomalies distribution detected on the test dataset. The experimental setup is outlined in Table 2. A. Exp-01, B.Exp-02, C. Exp-03, D.Exp-04.
</p>

## References
[PyCaret](https://pycaret.gitbook.io/docs/)
