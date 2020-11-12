#!/bin/bash
pip install gdown
gdown https://drive.google.com/uc?id=1F3GeBC3pXa0ofGhxYboN135BrlHV-Jtb -O yeti_data.zip
unzip yeti_data.zip
mkdir radar
mv yeti_data/*.png radar
rm -rf yeti_data
rm yeti_data.zip
