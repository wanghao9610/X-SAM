# :bookmark_tabs: Model Preparing

We provide an awesome [script](hfd.sh) to download models, thanks to [hfd](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f).

## X-SAM
```bash
cd $root_dir/docs
mkdir -p $root_dir/inits
chmod +x hfd.sh
alias hfd="$PWD/hfd.sh"

hfd hao9610/X-SAM --tools aria2c -x 8 --save_dir $root_dir/inits
```

## Phi-3-mini-4k-instruct
```bash
cd $root_dir/docs
mkdir -p $root_dir/inits
chmod +x hfd.sh
alias hfd="$PWD/hfd.sh"

hfd microsoft/Phi-3-mini-4k-instruct --tools aria2c -x 8 --save_dir $root_dir/inits
```

## Qwen3-4B-Instruct-2507
```bash
cd $root_dir/docs
mkdir -p $root_dir/inits
chmod +x hfd.sh
alias hfd="$PWD/hfd.sh"

hfd Qwen/Qwen3-4B-Instruct-2507 --tools aria2c -x 8 --save_dir $root_dir/inits
```

## siglip2-so400m-patch14-384
```bash
cd $root_dir/docs
mkdir -p $root_dir/inits
chmod +x hfd.sh
alias hfd="$PWD/hfd.sh"

hfd google/siglip2-so400m-patch14-384 --tools aria2c -x 8 --save_dir $root_dir/inits
```

## sam-vit-large
```bash
cd $root_dir/docs
mkdir -p $root_dir/inits
chmod +x hfd.sh
alias hfd="$PWD/hfd.sh"

hfd facebook/sam-vit-large --tools aria2c -x 8 --save_dir $root_dir/inits
```

## mask2former-swin-large-coco-panoptic
```bash
cd $root_dir/docs
mkdir -p $root_dir/inits
chmod +x hfd.sh
alias hfd="$PWD/hfd.sh"

hfd facebook/mask2former-swin-large-coco-panoptic --tools aria2c -x 8 --save_dir $root_dir/inits
```