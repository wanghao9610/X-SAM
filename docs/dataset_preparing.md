# :bookmark_tabs: Dataset Preparing

## Dataset Structure

```
datas
├── gcg_seg_data
│   ├── annotations
│   │   ├── train
│   │   └── val_test
│   └── images
│       ├── coco2014
│       ├── coco2017
│       ├── flickr30k
│       └── GranDf_HA_images
├── gen_seg_data
│   ├── ade20k
│   │   ├── ade20k_panoptic_train
│   │   ├── ade20k_panoptic_val
│   │   ├── annotations
│   │   ├── annotations_detectron2
│   │   └── images
│   ├── coco2017
│   │   ├── annotations
│   │   ├── panoptic_train2017
│   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   └── val2017
├── img_conv_data
│   ├── llava_images
│   │   ├── coco
│   │   ├── gqa
│   │   ├── ocr_vqa
│   │   ├── textvqa
│   │   └── vg
│   ├── LLaVA-Instruct-150K
│   └── LLaVA-Pretrain
│       └── 558k_images
├── inter_seg_data
│   ├── annotations
│   └── coco2017
├── LMUData
│   └── images
│       ├── AI2D_TEST
│       ├── MMBench
│       ├── MME
│       ├── POPE
│       └── SEEDBench_IMG
├── ov_seg_data
│   └── ade20k
├── rea_seg_data
│   ├── explanatory
│   ├── test
│   ├── train
│   └── val
├── ref_seg_data
│   ├── annotations
│   ├── images
│   │   ├── train2014
│   │   └── val2014
│   ├── refcoco
│   ├── refcoco+
│   └── refcocog
└── vgd_seg_data
    ├── annotations
    └── coco2017
```

## Image Segmentation Dataset

### 1. Generic Segmentation Dataset
```bash
cd $root_dir
mkdir -p datas/gen_seg_data/coco2017
export temp_data_dir=$root_dir/datas/gen_seg_data
# download coco2017 dataset
wget http://images.cocodataset.org/zips/train2017.zip -O $temp_data_dir/coco2017/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O $temp_data_dir/coco2017/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $temp_data_dir/coco2017/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip -O $temp_data_dir/coco2017/panoptic_annotations_trainval2017.zip

# unzip dataset and remove zip files
unzip $temp_data_dir/coco2017/train2017.zip -d $temp_data_dir/coco2017
unzip $temp_data_dir/coco2017/val2017.zip -d $temp_data_dir/coco2017
unzip $temp_data_dir/coco2017/annotations_trainval2017.zip -d $temp_data_dir/coco2017
unzip $temp_data_dir/coco2017/panoptic_annotations_trainval2017.zip -d $temp_data_dir/coco2017
unzip $temp_data_dir/coco2017/annotations/panoptic_train2017.zip -d $temp_data_dir/coco2017
unzip $temp_data_dir/coco2017/annotations/panoptic_val2017.zip -d $temp_data_dir/coco2017
rm $temp_data_dir/coco2017/train2017.zip $temp_data_dir/coco2017/val2017.zip $temp_data_dir/coco2017/annotations_trainval2017.zip $temp_data_dir/coco2017/panoptic_annotations_trainval2017.zip $temp_data_dir/coco2017/annotations/panoptic_train2017.zip $temp_data_dir/coco2017/annotations/panoptic_val2017.zip

# download coco2014 images
mkdir -p datas/gen_seg_data/coco2014
wget http://images.cocodataset.org/zips/train2014.zip -O $temp_data_dir/coco2014/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip -O $temp_data_dir/coco2014/val2014.zip
# unzip dataset
unzip $temp_data_dir/coco2014/train2014.zip -d $temp_data_dir/coco2014
unzip $temp_data_dir/coco2014/val2014.zip -d $temp_data_dir/coco2014
rm $temp_data_dir/coco2014/train2014.zip $temp_data_dir/coco2014/val2014.zip
# convert dataset
python $root_dir/xsam/xsam/tools/dataset_tools/prepare_ade20k_panoptic.py
python $root_dir/xsam/xsam/tools/dataset_tools/prepare_ade20k_semantic.py
python $root_dir/xsam/xsam/tools/dataset_tools/prepare_ade20k_instance.py

unset temp_data_dir
```

### 2. Open-Vocabulary(OV) Segmentation Dataset
```bash
cd $root_dir
mkdir -p datas/ov_seg_data
export temp_data_dir=$root_dir/datas/ov_seg_data
# download dataset
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O $temp_data_dir/ADEChallengeData2016.zip
# unzip dataset and rename the folder
unzip $temp_data_dir/ADEChallengeData2016.zip -d $temp_data_dir
mv $temp_data_dir/ADEChallengeData2016 $temp_data_dir/ade20k
# remove zip file
rm $temp_data_dir/ADEChallengeData2016.zip

unset temp_data_dir
```

### 3. Referring Segmentation Dataset
```bash
cd $root_dir
mkdir -p datas/ref_seg_data
mkdir -p datas/ref_seg_data/images
export temp_data_dir=$root_dir/datas/ref_seg_data
# download dataset
wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip -O $temp_data_dir/refcoco.zip
wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip -O $temp_data_dir/refcoco+.zip
wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip -O $temp_data_dir/refcocog.zip
# unzip dataset
unzip $temp_data_dir/refclef.zip -d $temp_data_dir
unzip $temp_data_dir/refcoco.zip -d $temp_data_dir
unzip $temp_data_dir/refcoco+.zip -d $temp_data_dir
unzip $temp_data_dir/refcocog.zip -d $temp_data_dir
rm $temp_data_dir/refclef.zip $temp_data_dir/refcoco.zip $temp_data_dir/refcoco+.zip $temp_data_dir/refcocog.zip    
unset temp_data_dir

# softlink coco2014 images
ln -s $root_dir/datas/gen_seg_data/coco2014 $temp_data_dir/images/coco2014

unset temp_data_dir
```

### 4. Reasoning Segmentation Dataset
Download the [Dataset](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy) (train.zip, val.zip, test.zip, explanatory/train.json) from Google Drive and put it in $root_dir/datas/rea_seg_data.
```bash
cd $root_dir
mkdir -p datas/rea_seg_data
export temp_data_dir=$root_dir/datas/rea_seg_data
mkdir -p $temp_data_dir/explanatory
# suppose you have downloaded the dataset and put them in $temp_data_dir as below structure
# rea_seg_data
# ├── train.zip
# ├── val.zip
# ├── test.zip
# └── train.json

# unzip dataset
unzip $temp_data_dir/train.zip -d $temp_data_dir
unzip $temp_data_dir/val.zip -d $temp_data_dir
unzip $temp_data_dir/test.zip -d $temp_data_dir
mv $temp_data_dir/train.json $temp_data_dir/explanatory/train.json
rm $temp_data_dir/train.zip $temp_data_dir/val.zip $temp_data_dir/test.zip

unset temp_data_dir
```

### 5. GCG Segmentation Dataset
Download the [Dataset](https://drive.usercontent.google.com/download?id=1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_&export=download&authuser=0&confirm=t&uuid=bb3fe3db-b08c-48f0-9280-2e56c0910987&at=AN8xHooqlXNOUCiIJYVHFMBLtmVn%3A1754293785835)(GranDf_HA_images.zip) from Google Drive and put it in $root_dir/datas/gcg_seg_data.
```bash
cd $root_dir
mkdir -p datas/gcg_seg_data datas/gcg_seg_data/images
export temp_data_dir=$root_dir/datas/gcg_seg_data
# download dataset
hfd MBZUAI/GranD-f --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
mv GranD-f $temp_data_dir/annotations
# unzip dataset
unzip $temp_data_dir/GranD-f_HA_images.zip -d $temp_data_dir/images
rm $temp_data_dir/GranD-f_HA_images.zip

# download flickr30k images
wget https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip -O $temp_data_dir/flickr30k-images.zip
unzip $temp_data_dir/flickr30k-images.zip -d $temp_data_dir/images
mv $temp_data_dir/images/flickr30k-images $temp_data_dir/images/flickr30k
rm $temp_data_dir/flickr30k-images.zip

# softlink coco2017 and coco2014 images
ln -s $root_dir/datas/gen_seg_data/coco2017 $temp_data_dir/images/coco2017
ln -s $root_dir/datas/ref_seg_data/coco2014 $temp_data_dir/images/coco2014

unset temp_data_dir
```

### 6. Interactive Segmentation Dataset
Download the [Dataset](https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0) (PSALM_data.zip) from Google Drive and put it in $root_dir/datas/inter_seg_data.
```bash
cd $root_dir
mkdir -p datas/inter_seg_data datas/inter_seg_data/annotations
export temp_data_dir=$root_dir/datas/inter_seg_data
# suppose you have downloaded the dataset and put them in $temp_data_dir as below structure
# inter_seg_data
# └── PSALM_data.zip

# unzip dataset
unzip $temp_data_dir/PSALM_data.zip -d $temp_data_dir
mv $temp_data_dir/PSALM_data/coco_interactive_train_psalm.json $temp_data_dir/PSALM_data/coco_interactive_val_psalm.json $temp_data_dir/annotations
ln -s $root_dir/datas/gen_seg_data/coco2017 $temp_data_dir/coco2017
rm -rf $temp_data_dir/PSALM_data $temp_data_dir/PSALM_data.zip

unset temp_data_dir
```

### 7. VGD Segmentation Dataset
Download the [Dataset](https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations) (vgdseg_annotations) from HuggingFace and put it in $root_dir/datas/vgd_seg_data.
```bash
cd $root_dir
mkdir -p datas/vgd_seg_data
export temp_data_dir=$root_dir/datas/vgd_seg_data
mkdir -p $temp_data_dir/images
# suppose you have downloaded the dataset and put them in $temp_data_dir as below structure
# vgd_seg_data
# ├── annotations
# |   ├──coco_vgdseg_train.json
# |   └──coco_vgdseg_val.json
# └── coco2017

# unzip dataset
unzip $temp_data_dir/vgd_seg_annotations.zip -d $temp_data_dir
mv $temp_data_dir/vgd_annotations $temp_data_dir/annotations
ln -s $root_dir/datas/gen_seg_data/coco2017 $temp_data_dir/coco2017
rm $temp_data_dir/vgd_seg_annotations.zip

unset temp_data_dir
```

## Image Conversation Dataset
We provide an awesome [script](hfd.sh) to download datasets, thanks to [hfd](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f).
### 1. LLaVA Training Dataset
```bash
cd $root_dir
mkdir -p datas/img_conv_data
export temp_data_dir=$root_dir/datas/img_conv_data
chmod +x hfd.sh
alias hfd="$PWD/hfd.sh"

hfd liuhaotian/LLaVA-Instruct-150K --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
hfd liuhaotian/LLaVA-Pretrain --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
ln -s $root_dir/datas/gen_seg_data/coco2017 $temp_data_dir/coco
mkdir $temp_data_dir/llava_images
# please prepare the GQA, OCR_VQA, TEXT_VQA, VG datasets and put them in $temp_data_dir/llava_images as below structure
# llava_images
# ├── coco
# ├── gqa
# ├── ocr_vqa
# ├── text_vqa
# └── vg

unset temp_data_dir
```

### 2. VLM Evaluation Dataset
```bash
data_dir=$root_dir/datas
export LMUData="$data_dir/LMUData"
mkdir -p $LMUData

# vlmeval will download datasets automatically
bash $root_dir/runs/run.sh \
    --modes vlmeval \
    --config xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune.py
```