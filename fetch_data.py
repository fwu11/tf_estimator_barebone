# This file fetches data from the internet
# PASCAL VOC 2012 dataset
# PASCAL augmented dataset 
# 10582 (train aug)
# 1449 (val)


# augmented PASCAL VOC
cd dataset 
wget --content-disposition https://uc630473502ca9f7d183a61baa95.dl.dropboxusercontent.com/cd/0/get/AVCo485cw02Qbs8pvxvsQJ9dckZNo5kPnZuF5uospf3pbdIyv9KPqeNYpU6odEtLoV5MukGzTw5ZsF1WQjmbjVag_VBx2D1aJmBmPctewecF77c7TUhnnpgUZpLeIKReuBWORHJitCfJxqUE7WnIAx4GEvV_yiqFJ7-LgJHkRAPidojsBVLgb_0a6PyXS9NpPf4/file?_download_id=36974684572386053159328901397914718847396332166223670769349598242
unzip SegmentationClassAug.zip 
rm -r __MACOSX
rm -r SegmentationClassAug.zip 
# down load trainaug.txt
wget https://gist.githubusercontent.com/sun11/2dbda6b31acc7c6292d14a872d0c90b7/raw/5f5a5270089239ef2f6b65b1cc55208355b5acca/trainaug.txt


# original PASCAL VOC 2012 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB 
tar -xvf VOCtrainval_11-May-2012.tar 
rm -r VOCtrainval_11-May-2012.tar 



