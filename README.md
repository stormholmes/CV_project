# CV project based on GeCo
This repo is based on the code of [A novel unified architecture for low-shot counting by detection and segmentation](https://github.com/jerpelhan/GeCo) 

Please refer to GeCo_README.md for the environment.

To run the code, you need to download DATA.zip from https://drive.google.com/drive/folders/1_0qdENRSc7pdHvUIxG3IKDTm9wb-1Mq9?usp=drive_link and unzip it in to the DATA folder. And download GeCo-base.pth, GeCoLF.pth and sam_hq_vit_h.pth from https://drive.google.com/drive/folders/1_0qdENRSc7pdHvUIxG3IKDTm9wb-1Mq9?usp=drive_link and put them into the MODEL folder.

The test.sh can be run directly, while the pretrain1.sh and train1.sh are the modified versions of pretrain.sh and train.sh for the HKUST superPOD.

The visualization results for reference are in ./vis.
