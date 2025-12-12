"""
model_downloader.py
Download datasets and models from huggingface hub and other places
"""
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import datasets
import utilities


def download_models(the_config: dict) -> None:
    """load models and tokenizers"""
    for i_name in the_config['model2path']:
        """pythia for scaling"""
        if any(substring in i_name for substring in ['opt', 'qwen']):
            print(f"Working with {the_config['model2path'][i_name]} checkpoints...")
            tokenizer = AutoTokenizer.from_pretrained(the_config['model2path'][i_name], cache_dir=the_config['location']['models'])
            model = AutoModelForCausalLM.from_pretrained(the_config['model2path'][i_name], cache_dir=the_config['location']['models'])
        elif any(substring in i_name for substring in ['flant5']):
            print(f"Working with {i_name} checkpoints...")
            tokenizer = AutoTokenizer.from_pretrained(the_config['model2path'][i_name], cache_dir=the_config['location']['models'])
            model = AutoModelForSeq2SeqLM.from_pretrained(the_config['model2path'][i_name], cache_dir=the_config['location']['models'])

 
def download_datasets(the_config: dict, download_mode:str='reuse_dataset_if_exists') -> None:
    """
    download dataset
    download_mode: ['reuse_dataset_if_exists' | 'reuse_cache_if_exists' | 'force_redownload']
    """
    for i_name in the_config['dataset2path']:
        print(f"Downloading {i_name} dataset.")
        if isinstance(the_config['dataset2path'][i_name], str):
            data = datasets.load_dataset(the_config['dataset2path'][i_name],
                                         cache_dir=the_config['location']['datasets'],
                                         download_mode=download_mode,
                                         trust_remote_code=True)
        elif isinstance(the_config['dataset2path'][i_name], list):
            tmp = tuple(the_config['dataset2path'][i_name])
            data = datasets.load_dataset(*tmp,
                                         cache_dir=the_config['location']['datasets'],
                                         download_mode=download_mode,
                                         trust_remote_code=True)