# Install requirements
pip install -r requirements.txt

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Download COCO dataset
cd data
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip downloaded files
unzip val2017.zip
unzip test2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip
