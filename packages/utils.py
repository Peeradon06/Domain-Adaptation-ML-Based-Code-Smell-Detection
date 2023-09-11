import os
from os import path
import sys
sys.path.append('../')
from config.definitions import FONTANA_DIR, MLCQ_DIR, MAPPED_DATASET_DIR
import pandas as pd
import numpy as np

def create_report(models_dict, optimize_ct_dict=None, report_name="/unnamed.csv"):
    """ Create report 

    Parameters 
    ----------
    models_dict : dict
        Dictionary contains model's dictionary, for example, 
            baseline_model = {"GOD_CLASS", [DT_classifier, RF_classifier]}
    optimize_ct_dict : dict
        Dictionary contains optimization time of [DT_classifier, RF_classifier] for each smell
    report_path : str
        Path and name of the report to be export in ./reports 
    
    Example
    -------
    >>> create_report(bo_models, bo_ct, report_name="/mlcq/bo_0.csv")
    """

    # validation metrics
    acc = []
    pre = []
    rec = []
    f1 = []
    idx = []
    roc = []
    training_time = []
    
    # prediciton metrics 
    pred_acc = []
    pred_pre = []
    pred_rec = []
    pred_f1 = []
    pred_roc = []
    
    # optimization time 
    optimize_time = []
    total_time = []
    
    smells = models_dict.keys()
    
    for i in range(2):
        for k in smells:
            model = models_dict[k][i]
            
            idx.append(model.model_name)      

            # validataion set
            acc.append(model.accuracy)
            pre.append(model.precision)
            rec.append(model.recall)
            f1.append(model.f1)
            roc.append(model.roc_auc)
            training_time.append(model.training_time)

            # test set
            pred_acc.append(model.pred_accuracy)
            pred_pre.append(model.pred_precision)
            pred_rec.append(model.pred_recall)
            pred_f1.append(model.pred_f1)
            pred_roc.append(model.pred_roc)
        
        # optimization time 
            if optimize_ct_dict:
                opt_compute_time = optimize_ct_dict[k][i]
                optimize_time.append(opt_compute_time)
                
            else:
                opt_compute_time = 0
                optimize_time.append(opt_compute_time)
            
            total_time.append(model.training_time + opt_compute_time)
                    
    data = {
        'Accuracy' : acc,
        'Precision' : pre,
        'Recall' : rec,
        'F1' : f1,
        'ROC' : roc,
        'Test_Accuracy' : pred_acc,
        'Test_Precision' : pred_pre,
        'Test_Recall' : pred_rec,
        'Test_F1' : pred_f1,
        'Test_ROC' : pred_roc,
        'Training_time' : training_time,
        'Optimize_time' : optimize_time,
        'Total_time' : total_time
    }
    
    df = pd.DataFrame(data, index=idx)
    
    # check for exist directory 
    dirs = ['', 'qc', 'mlcq', 'transfer']
    for d in dirs:
        dir_path = '../reports/{}'.format(d)
        if path.isdir(dir_path) == False:
            os.makedirs(dir_path)

    try:
        df.to_csv('../reports'+report_name)
        print('Report exported ! at {}'.format(report_name))
    
    except Exception as e:
        print('Error while exporting the report ', e)

def read_mapped_dataset(data="QC"):
    """ Read the mapped datasets

    Parameters 
    ----------
    data : string
        String for selecting datasets between ["QC", "MLCQ"]
    
    Return
    ------
    datasets : dict 
        Dictonary contains four code smells which keys are the name and values are dataframe
    
    Example
    ------
    >>> mapped_qc = read_mapped_dataset('QC')
    """

    print(f'Loading Mapped {data} dataset . . .')
    
    if data.upper() == "QC":
        
        try:
            data_class = pd.read_csv(MAPPED_DATASET_DIR+"mapped_data_class.csv")
            feature_envy = pd.read_csv(MAPPED_DATASET_DIR+"mapped_feature_envy.csv")
            god_class = pd.read_csv(MAPPED_DATASET_DIR+"mapped_god_class.csv")
            long_method = pd.read_csv(MAPPED_DATASET_DIR+"mapped_long_method.csv")

            datasets = {
                "Data_Class": data_class,
                "Feature_Envy": feature_envy,
                "God_Class": god_class,
                "Long_Method": long_method
            }

            print("Done all mapped QC datasets are loaded !")
            
            return datasets

        except Exception as e:
            print("!== Exception while reading Mapped QC dataset ==!\n")
            print(e)

    if data.upper() == "MLCQ":
        
        try:
            data_class = pd.read_csv(MAPPED_DATASET_DIR+"mapped_mlcq_data_class.csv")
            feature_envy = pd.read_csv(MAPPED_DATASET_DIR+"mapped_mlcq_feature_envy.csv")
            god_class = pd.read_csv(MAPPED_DATASET_DIR+"mapped_mlcq_god_class.csv")
            long_method = pd.read_csv(MAPPED_DATASET_DIR+"mapped_mlcq_long_method.csv")

            datasets = {
                "Data_Class": data_class,
                "Feature_Envy": feature_envy,
                "God_Class": god_class,
                "Long_Method": long_method
            }
            
            print("Done all mapped MLCQ datasets are loaded !")
            
            return datasets
        
        except Exception as e:
            print("!== Exception while reading Mapped MLCQ dataset ==!\n", e)
    else:
        print("Can't found dataset, please try insert 'QC' or 'MLCQ' in data parameter")

def export_report(models_dict, dir="/reports/baseline.csv"):
    """ Export models result reports
    
    Parameters 
    ----------
    models_dict : dict
        Dictionary contains model's dictionary, for example, 
            baseline_model = {"GOD_CLASS", [DT_classifier, RF_classifier]}
    dir : string 
        Path of the exported reports (always export in .csv files)
    
    Example
    -------
    >>> export_report(baseline_models, dir='/qc-baseline-result/baseline.csv')
    """

    # validation metrics
    acc = []
    pre = []
    rec = []
    f1 = []
    idx = []
    roc = []
    training_time = []
    
    # prediciton metrics 
    pred_acc = []
    pred_pre = []
    pred_rec = []
    pred_f1 = []
    pred_roc = []

    # loop through each smell dataset
    for k,v in models_dict.items():
        # each model (DT and RF)
        for model in models_dict[k]:
            # store the name as an index
            idx.append(model.model_name)      

            # validataion set
            acc.append(model.accuracy)
            pre.append(model.precision)
            rec.append(model.recall)
            f1.append(model.f1)
            roc.append(model.roc_auc)
            training_time.append(model.training_time)

            # test set
            pred_acc.append(model.pred_accuracy)
            pred_pre.append(model.pred_precision)
            pred_rec.append(model.pred_recall)
            pred_f1.append(model.pred_f1)
            pred_roc.append(model.pred_roc)
    
    data = {
    'Accuracy' : acc,
    'Precision' : pre,
    'Recall' : rec,
    'F1' : f1,
    'ROC' : roc,
    'Test_Accuracy' : pred_acc,
    'Test_Precision' : pred_pre,
    'Test_Recall' : pred_rec,
    'Test_F1' : pred_f1,
    'Test_ROC' : pred_roc,
    'Training_time' : training_time,
    }
    
    df = pd.DataFrame(data, index=idx)
    
    # check for exist directory 
    dirs = ['', 'qc', 'mlcq', 'cross']
    for d in dirs:
        dir_path = '../reports/{}'.format(d)
        if path.isdir(dir_path) == False:
            os.makedirs(dir_path)

    try:
        df.to_csv('../reports'+dir)
    except Exception as e:
        print('Error while exporting the report ', e)
    
    print('Report exported ! at /reports{}'.format(dir))