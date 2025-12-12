# FIRST(before all, even torch) Cuda visibility
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                                        
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   

import utilities
import torch
import model_downloader
import process_dataset
from fmodel import ModelTraining
from gens import GenerateText


if __name__ =="__main__":
    """"""
    jrnl = utilities.Journaling()

    """====generate with saved model"""
    jrnl.start(disable_log=False, check_cli=True)
    the_gentext = GenerateText(the_journal=jrnl)
    the_gentext.generate_n_store()