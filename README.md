# Divergence-Aware Selective Data-to-Text Generation

This repository contains the implementation for “Divergence-Aware Selective Data-to-Text Generation with LLMs for Factual Consistency.”  
For any queries, please feel free to contact the corresponding author.

## File Structure and Description

* [sr_divergence.py](sr_divergence.py): Contains three measures for assessing source–reference divergence.  
* [fmodel.py](fmodel.py): Includes all modelling components.  
* [main.py](main.py): Entry script for training and obtaining dataset statistics.  
* [main-gen.py](main-gen.py): Script for LLM-based data-to-text generation.  
* [settings/](settings/): Directory for configuration and setup files.  
    * [settings/credentials.toml](settings/credentials.toml): Credentials file.  
    * [settings/main_config.toml](settings/main_config.toml): Main configuration settings.  
* [README.md](README.md): Overview of the project, file descriptions, and authorship details.

## Author(s)

* [Joy Mahapatra](joymahapatra90@gmail.com) (Corresponding author)
* [Utpal Garain](https://www.isical.ac.in/~utpal/)

## Acknowledgments

This research is partially supported by the Indo-French Centre for the Promotion of Advanced Research (IFCPAR/CEFIPRA) under CSRP Project No. 6702-2.

