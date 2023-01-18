# Unsupervised-Motion-Artefact-Detection



## Data 

### MR-ART


### IXI
separated into files (see zip file)


## Models
### DeepSVDD




Example command for IXI dataset, training on 10 MRIs
```
python main.py ixi  MVTEC_LeNet <path_to_write_out_log_files> <path_to_data> --data_split_path ~/motion/dfs/ --eval_epoch 1  --device cuda:2 --n  10  --objective one-class --seed 1001 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True  --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 1
```

The model pretrains before training. The model is evaluated on the test set after each epoch. 

### Output

The output for the above command is 
The epoch with the best result will write out a file with /final_data_n_10_seed_1001' with columns, 

'output' - anomaly score (based on mean anomaly scores of slices) 
'output1' - anomaly score (based on max anomaly score of slices)
'label' - 0 for normals, 1s for anomalies. 
'label_sev' - ignore for IXI dataset, only applicable to 'MR-ART', 0 for normal, 1 for medium quality and 2 for good quality.
'pred' - converting 'output' to binary value of 0 and 1 using a threshold based on the class balance
'pred2' - converting 'output1' to binary value of 0 and 1 using a threshold based on the class balance

The log file shows after each time it is evaluated on the test set (after each epoch)
'Test set AUC based on mean anomaly score per volume:'
Test set F1 based on mean anomaly score per volume:'
'Test set Balanced Accuarcy based on mean anomaly score per volume:'
'Test set AUC based on max anomaly score per volume:'
'Test set F1 based on max anomaly score per volume:'
'Test set Balanced Accuracy based on max anomaly score per volume:'
'Average inference time per batch'

     
