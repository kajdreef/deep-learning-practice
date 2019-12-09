# Week 9
For both versions PlaidML was installed to speed up the training of the models. 
The environment was setup as following (for plaidml-setup the AMD-gpu on the macbook pro was used and the config need to be saved)

brew cask install anaconda
conda create -n plaidml python=3.7 anaconda
source activate plaidml
conda install -n pydot -y
pip install plaidml-keras
plaidml-setup


## ThirtyFive.py
python3 ThirtyFive.py ../pride-and-prejudice.txt


## stop_words_elimination.py
python3 stop_words_elimination.py ../pride-and-prejudice.txt