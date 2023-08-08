# Video Inpainting Detection
This is the README file for Video Inpainting Detection project. Current workflow is as follows:

```mermaid
flowchart LR
A[Videos] -->|SAT| B[Annotations]
A --> C(Inpainting MAT/E2FGVI)
B --> C
C --> D(VIS vistr/IFC)
```

>Notice: when running a python program, please cd to its directory first.


## 1. Folder Info
```
|--E2FGVI                    // Inpainting method E2FGVI
|--MAT                       // Inpainting method MAT
|--tools                     // some useful tools
|--IFC                       // Video Instance Segmentation IFC, not success
|--vistr                     // Video Instance Segmentation vistr
```
Reference: 
* [E2FGVI](https://github.com/MCG-NKU/E2FGVI)
* [MAT](https://github.com/fenglinglwb/MAT)
* [IFC](https://github.com/sukjunhwang/IFC)
* [vistr](https://github.com/Epiphqny/VisTR)
## 2. Video Inpainting
1. Use Segment-and-Track-Anything(SAT) to obtain masks of object that you want to inpaint.
2. Use MAT or E2FGVI to do video inpainting.

### MAT
Training:

1. Put data into `datasets/cars_train` and `datasets/cars_val`
2. Start training:
```
python train.py \
    --outdir=output_path \
    --gpus=8 \                          // modify this as needed
    --batch=32 \                        // modify this as needed
    --metrics=fid36k5_full \
    --data=training_data_path \         // specify as datasets/cars_train
    --data_val=val_data_path \          // specify as datasets/cars_val
    --dataloader=datasets.dataset_512.ImageFolderMaskDataset \
    --mirror=True \
    --cond=False \
    --cfg=places512 \
    --aug=noaug \
    --generator=networks.mat.Generator \
    --discriminator=networks.mat.Discriminator \
    --loss=losses.loss.TwoStageLoss \
    --pr=0.1 \
    --pl=False \
    --truncation=0.5 \
    --style_mix=0.5 \
    --ema=10 \
    --lr=0.001
```
Testing:
```
cd MAT
python generate_image.py --network model_path --dpath data_path --mpath mask_path --outdir out_path
```
### E2FGVI
Training:

1. Prepare dataset as follows:
```
datasets
  |--cars
  |  |--JPEGImages
  |  |  |--<video_name>.zip      // the zip file contains frames in the video
  |  |  |--<video_name>.zip
  |  |--test_masks
  |  |  |--<video_name>
  |  |  |  |--00000.png
  |  |  |  |--00001.png
  |  |--train.json               // a dict of {video_name:number_of_frames}
  |  |--test.json
```
2. Modify the `train_data_loader->name` to `cars` in `configs/train_e2fgvi_hq.json` or `configs/train_e2fgvi.json`.
3. Start training (select one):
```
 python train.py -c configs/train_e2fgvi.json
 python train.py -c configs/train_e2fgvi_hq.json
```
Testing:
```
cd E2FGVI
# Using this command to output a 720p video
python test.py --model e2fgvi_hq --video <video_path> --mask <mask_path>  --ckpt release_model/your_model.pth --set_size --width 1280 --height 720
```

## 3. Video Inpainting Detection
Here we use vistr for detection. Follow original repo for installment guide.
1. Data Preparation:

Place your inpainting images and masks in folder `data/cars` as follows:
```
├── data
│   ├── train
|   |   ├── Annotations
|   |   ├── JPEGImages
│   ├── val
|   |   ├── Annotations
|   |   ├── JPEGImages
│   ├── annotations
``` 
then run:
```bash
cd data/cars
python create_json.py
```
>Notice: check the `create_json.py` file and modify parameters as needed.

2. Start Training:
```
python -m torch.distributed.launch \
    --nproc_per_node=8 \                    // num of gpus used
    --use_env main.py \
    --backbone resnet101/50 \               // use backbone resnet50 or resnet101
    --ytvos_path /path/to/ytvos \           // data/cars
    --masks \
    --pretrained_weights /path/to/pretrained_path // pretraine/***.pth
```

Inference:
```
python inference.py \
    --masks \
    --model_path /path/to/model_weights \
    --backbone resnet101/50 \
    --img_path /path/to/test/images \        // data/cars/val/JPEGImages
    --ann_path /path/to/test/annotation \    // data/cars/annotation/instance_val_sub.json
    --save_path /path/to/results.json \
    --dataset_file cars
```
Results are stored in /path/to/results.json, convert it to images by using
```
python json2img.py \
    --json /path/to/results.json \
    --img /path/to/test/images \             // data/cars/val/JPEGImages
    --ann /path/to/test/annotation \         // data/cars/annotation/instance_val_sub.json
    --out /path/to/output \                  // data/cars/val/inference

```
