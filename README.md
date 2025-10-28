# Install & Setup
## On Linux/MacOS
```
conda create -n transar python=3.8
conda activate transar
pip install -r requirements.txt
```
Additionally you can check torch is working with Metal (for MacOS) backend with :  
```
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
``` 

# Pretraining 
## Dataset
[Capella Space OpenData](https://felt.com/map/Capella-Space-Open-Data-bB24xsH3SuiUlpMdDbVRaA?loc=0,-20.5,1.64z) was used as dataset for pretraining.
SAR images must be preprocessed as single precision tiles (.npy) before training.
The following command allows to chip images from a given path into 512x512 chips:  
```
python data/chip_capella.py /path/to/sar_images --chip_size 512
``` 

# Fine-tuning 
## Dataset
[SARDet-100k](https://www.kaggle.com/datasets/greatbird/sardet-100k) opensource dataset was used supervised fine-tuning and for the experiments in this repo.