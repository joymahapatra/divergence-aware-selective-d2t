# FIRST(before all, even torch) Cuda visibility
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                                        
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   

import utilities
import torch
import model_downloader
import process_dataset
from fmodel import ModelTraining
from sr_divergence import SRDivergence


if __name__ =="__main__":
    """"""
    jrnl = utilities.Journaling()


    """====access model_downloader.py"""
    # jrnl.start(disable_log=True)
    # model_downloader.download_models(the_config=jrnl.main_config)
    # model_downloader.download_datasets(the_config=jrnl.main_config)


    """====check porocess_datset.py"""
    # jrnl.start(disable_log=True)
    # all_datasets = jrnl.main_config['dataset2class']
    # for i_dataset in ['wikitabletext']:
    #     tmp_dc = getattr(process_dataset, jrnl.main_config['dataset2class'][i_dataset])(the_config=jrnl.main_config)
    #     tmp_dd = tmp_dc.build()
    #     for i_partition in tmp_dd:
    #         print(f"In {i_partition}-partition of {i_dataset}-dataset.")
    #         random_row = 15
    #         for k, field in enumerate(tmp_dd[i_partition][random_row].keys()):
    #             print(f"\t{k}. {field}-contain ---> {tmp_dd[i_partition][random_row][field]}")        
    #     input("waiting..")


    """====run model"""
    jrnl.start(check_cli=True)
    mt_run = ModelTraining(the_journal=jrnl)
    mt_run.run_training()
