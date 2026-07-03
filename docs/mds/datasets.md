# :bookmark_tabs: Datasets

## Structure

```
datas
├── img_chat
│   └── llava
│       ├── LLaVA-CC3M-Pretrain-595K
│       ├── llava_images
│       ├── LLaVA-Instruct-150K
│       └── LLaVA-Pretrain
├── img_gcgseg
│   └── grand_f
│       ├── annotations
│       │   ├── train
│       │   └── val_test
│       └── images
│           ├── coco2014 -> ../../../img_genseg/coco2014
│           ├── coco2017 -> ../../../img_genseg/coco2017
│           ├── flickr30k
│           └── GranDf_HA_images
├── img_genseg
│   ├── coco2017
│   │   ├── annotations
│   │   ├── panoptic_semseg_train2017
│   │   ├── panoptic_semseg_val2017
│   │   ├── panoptic_train2017
│   │   ├── panoptic_val2017
│   │   ├── stuff_train2017_pixelmaps
│   │   ├── stuff_val2017_pixelmaps
│   │   ├── test2017
│   │   ├── train2014 -> ../coco2014/train2014
│   │   ├── train2017
│   │   ├── val2014 -> ../coco2014/val2014
│   │   └── val2017
│   └── coco2014
│       ├── annotations
│       ├── test2014
│       ├── train2014
│       └── val2014
├── img_intseg
│   └── coco_int
│       ├── annotations
│       └── coco2017 -> ../../img_genseg/coco2017
├── img_ovseg
│   └── ade20k
│       ├── ade20k_instance_catid_mapping.txt
│       ├── ade20k_instance_imgCatIds.json
│       ├── ade20k_instance_train.json
│       ├── ade20k_instance_val.json
│       ├── ade20k_panoptic_train
│       ├── ade20k_panoptic_train.json
│       ├── ade20k_panoptic_val
│       ├── ade20k_panoptic_val.json
│       ├── annotations
│       ├── annotations_detectron2
│       ├── annotations_instance
│       ├── images
│       ├── objectInfo150.txt
│       └── sceneCategories.txt
├── img_reaseg
│   └── lisa
│       ├── explanatory
│       ├── test
│       ├── train
│       └── val
├── img_refseg
│   └── refcocos
│       ├── annotations
│       ├── grefcoco
│       ├── images
│       │   └── train2014 -> ../../../img_genseg/coco2014/train2014
│       ├── refclef
│       ├── refcoco
│       ├── refcoco+
│       ├── refcocog
│       └── refcocop -> refcoco+
├── img_vgdseg
│   └── coco_vgd
│       ├── annotations
│       └── coco2017 -> ../../img_genseg/coco2017
├── LMUData
│   ├── AI2D_TEST.tsv
│   ├── GQA_TestDev_Balanced.tsv
│   ├── images
│   │   ├── AI2D_TEST
│   │   ├── GQA_TestDev_Balanced
│   │   ├── MLVU_MCQ
│   │   ├── MLVU_OpenEnded
│   │   ├── MMBench
│   │   ├── MMBench_V11
│   │   ├── MME
│   │   ├── POPE
│   │   ├── ScienceQA_TEST
│   │   ├── ScienceQA_VAL
│   │   ├── SEEDBench_IMG
│   ├── MMBench_DEV_EN.tsv
│   ├── MMBench_DEV_EN_V11.tsv
│   ├── MME.tsv
│   ├── POPE_local.tsv
│   ├── POPE.tsv
│   ├── ScienceQA_TEST.tsv
│   ├── ScienceQA_VAL.tsv
│   └── SEEDBench_IMG.tsv
```

## HFD Downloader Setting

We provide a custom downloader [`hfd`](../srcs/tools/hfd.sh) for downloading datasets, you can use it to download datasets from Hugging Face.

```bash
chmod +x $PROJ_HOME/srcs/tools/hfd.sh
alias hfd="$PROJ_HOME/srcs/tools/hfd.sh"
```

## Image Segmentation Datasets

### 1. Image Generic Segmentation Datasets

* COCO Dataset for Image Generic Segmentation (Semantic, Instance, Panoptic)

    Please refer to the following steps to download and process COCO dataset.
    ```bash
    cd "$PROJ_HOME"
    mkdir -p datas/img_genseg/coco2017
    export temp_data_dir=$PROJ_HOME/datas/img_genseg
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
    mkdir -p datas/img_genseg/coco2014
    wget http://images.cocodataset.org/zips/train2014.zip -O $temp_data_dir/coco2014/train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip -O $temp_data_dir/coco2014/val2014.zip
    # unzip dataset
    unzip $temp_data_dir/coco2014/train2014.zip -d $temp_data_dir/coco2014
    unzip $temp_data_dir/coco2014/val2014.zip -d $temp_data_dir/coco2014
    rm $temp_data_dir/coco2014/train2014.zip $temp_data_dir/coco2014/val2014.zip

    unset temp_data_dir
    ```

### 2. Image Open-Vocabulary Segmentation Datasets

* ADE20K Dataset for Image Open-Vocabulary Segmentation

    Please refer to the following steps to download and process ADE20K dataset.
    ```bash
    cd "$PROJ_HOME"
    mkdir -p datas/img_ovseg
    export temp_data_dir=$PROJ_HOME/datas/img_ovseg
    # download dataset
    wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O $temp_data_dir/ADEChallengeData2016.zip
    # unzip dataset and rename the folder
    unzip $temp_data_dir/ADEChallengeData2016.zip -d $temp_data_dir
    mv $temp_data_dir/ADEChallengeData2016 $temp_data_dir/ade20k
    # remove zip file
    rm $temp_data_dir/ADEChallengeData2016.zip
    # convert dataset
    python $PROJ_HOME/x2sam/x2sam/tools/dataset_tools/prepare_ade20k_panoptic.py
    python $PROJ_HOME/x2sam/x2sam/tools/dataset_tools/prepare_ade20k_semantic.py
    python $PROJ_HOME/x2sam/x2sam/tools/dataset_tools/prepare_ade20k_instance.py

    unset temp_data_dir
    ```

### 3. Image Referring Segmentation Datasets

* RefCOCO/+/g Datasets for Image Referring Segmentation

    Please refer to the following steps to download and process RefCOCO/+/g datasets.
    ```bash
    cd "$PROJ_HOME"
    mkdir -p datas/img_refseg/refcocos/images
    export temp_data_dir=$PROJ_HOME/datas/img_refseg/refcocos
    # download dataset
    wget https://web.archive.org/web/20220413011631/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip -O $temp_data_dir/refclef.zip
    wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip -O $temp_data_dir/refcoco.zip
    wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip -O $temp_data_dir/refcoco+.zip
    wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip -O $temp_data_dir/refcocog.zip
    # unzip dataset
    unzip $temp_data_dir/refclef.zip -d $temp_data_dir
    unzip $temp_data_dir/refcoco.zip -d $temp_data_dir
    unzip $temp_data_dir/refcoco+.zip -d $temp_data_dir
    unzip $temp_data_dir/refcocog.zip -d $temp_data_dir
    rm $temp_data_dir/refclef.zip $temp_data_dir/refcoco.zip $temp_data_dir/refcoco+.zip $temp_data_dir/refcocog.zip

    # softlink coco2014 images
    ln -s $PROJ_HOME/datas/img_genseg/coco2014/train2014 $temp_data_dir/images/train2014

    unset temp_data_dir
    ```
* gRefCOCO Datasets for Image Referring Segmentation

    Please refer to the following steps to download and process gRefCOCO datasets.
    ```bash
    cd "$PROJ_HOME"
    mkdir -p datas/img_refseg/refcocos/grefcoco
    export temp_data_dir=$PROJ_HOME/datas/img_refseg/refcocos/grefcoco
    cd $temp_data_dir
    hfd gRefCOCO/gRefCOCO --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mv $temp_data_dir/gRefCOCO/* $temp_data_dir
    rm -rf $temp_data_dir/gRefCOCO

    unset temp_data_dir
    ```

### 4. Image Reasoning Segmentation Datasets

* Lisa Dataset for Image Reasoning Segmentation

    Please refer to the [Lisa Dataset](https://github.com/JIA-Lab-research/LISA) to [download the dataset](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy), then refer to the following steps to process the dataset.

    ```bash
    cd "$PROJ_HOME"
    mkdir -p datas/img_reaseg/lisa
    export temp_data_dir=$PROJ_HOME/datas/img_reaseg/lisa
    mkdir -p $temp_data_dir/explanatory
    # suppose you have downloaded the dataset and put them in $temp_data_dir as below structure
    # img_reaseg
    # └── lisa
    #     ├── train.zip
    #     ├── val.zip
    #     ├── test.zip
    #     └── explanatory
    #         └── train.json

    # unzip dataset
    unzip $temp_data_dir/train.zip -d $temp_data_dir
    unzip $temp_data_dir/val.zip -d $temp_data_dir
    unzip $temp_data_dir/test.zip -d $temp_data_dir
    mv $temp_data_dir/train.json $temp_data_dir/explanatory/train.json
    rm $temp_data_dir/train.zip $temp_data_dir/val.zip $temp_data_dir/test.zip

    unset temp_data_dir
    ```

### 5. Image GCG Segmentation Datasets

* GranD-f Dataset for Image GCG Segmentation
    Download the [Dataset](https://drive.usercontent.google.com/download?id=1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_&export=download&authuser=0&confirm=t&uuid=bb3fe3db-b08c-48f0-9280-2e56c0910987&at=AN8xHooqlXNOUCiIJYVHFMBLtmVn%3A1754293785835)(GranDf_HA_images.zip) from Google Drive and put it in `$PROJ_HOME/datas/img_gcgseg/grand_f`.
    ```bash
    cd "$PROJ_HOME"
    mkdir -p datas/img_gcgseg/grand_f/images
    export temp_data_dir=$PROJ_HOME/datas/img_gcgseg/grand_f
    # download dataset
    hfd MBZUAI/GranD-f --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mv $temp_data_dir/GranD-f $temp_data_dir/annotations
    # unzip dataset
    unzip $temp_data_dir/GranD-f_HA_images.zip -d $temp_data_dir/images
    rm $temp_data_dir/GranD-f_HA_images.zip

    # download flickr30k images
    wget https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip -O $temp_data_dir/flickr30k-images.zip
    unzip $temp_data_dir/flickr30k-images.zip -d $temp_data_dir/images
    mkdir -p $temp_data_dir/images/flickr30k/images
    mv $temp_data_dir/images/flickr30k-images $temp_data_dir/images/flickr30k/images/train
    rm $temp_data_dir/flickr30k-images.zip

    # softlink coco2017 and coco2014 images
    ln -s $PROJ_HOME/datas/img_genseg/coco2017 $temp_data_dir/images/coco2017
    ln -s $PROJ_HOME/datas/img_genseg/coco2014 $temp_data_dir/images/coco2014

    unset temp_data_dir
    ```

### 6. Image Interactive Segmentation Datasets

* COCO-Interactive Dataset for Image Interactive Segmentation

    Please refer to the [COCO-Interactive Dataset](https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0) to [download the dataset](https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0) (PSALM_data.zip), then refer to the following steps to process the dataset.
    
    ```bash
    cd "$PROJ_HOME"
    mkdir -p datas/img_intseg/coco_int
    export temp_data_dir=$PROJ_HOME/datas/img_intseg/coco_int
    mkdir -p $temp_data_dir/annotations
    # download dataset
    wget https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0 -O $temp_data_dir/PSALM_data.zip
    # unzip dataset
    unzip $temp_data_dir/PSALM_data.zip -d $temp_data_dir
    mv $temp_data_dir/PSALM_data/coco_interactive_train_psalm.json $temp_data_dir/PSALM_data/coco_interactive_val_psalm.json $temp_data_dir/annotations
    ln -s $PROJ_HOME/datas/img_genseg/coco2017 $temp_data_dir/coco2017
    rm -rf $temp_data_dir/PSALM_data $temp_data_dir/PSALM_data.zip

    unset temp_data_dir
    ```

### 7. Image VGD Segmentation Datasets

* COCO-VGD Dataset for Image VGD Segmentation
    
    Please refer to the [COCO-VGD Dataset](https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations) to [download the dataset](https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations) (vgdseg_annotations), then refer to the following steps to process the dataset.
    
    ```bash
    cd "$PROJ_HOME"
    mkdir -p datas/img_vgdseg/coco_vgd
    export temp_data_dir=$PROJ_HOME/datas/img_vgdseg/coco_vgd
    mkdir -p $temp_data_dir/annotations
    # download dataset
    wget https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations -O $temp_data_dir/vgdseg_annotations.zip
    # unzip dataset
    unzip $temp_data_dir/vgdseg_annotations.zip -d $temp_data_dir
    mv $temp_data_dir/vgdseg_annotations/* $temp_data_dir/annotations
    ln -s $PROJ_HOME/datas/img_genseg/coco2017 $temp_data_dir/coco2017
    rm -rf $temp_data_dir/vgdseg_annotations $temp_data_dir/vgdseg_annotations.zip

    unset temp_data_dir
    ```

* Image Chat & Video Chat Benchmark Datasets

    `VLMEvalKit` will automatically download the image chat and video chat benchmark datasets for evaluation.
