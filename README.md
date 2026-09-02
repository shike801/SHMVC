# SHMVC: Semantic-Guided Hierarchical Multi-View Contrastive Clustering for Hyperspectral Images



## Requirements

CUDA Version: 11.3

torch: 1.11.0

Python: 3.8.10



## Dataset

The dataset directory should look like this:

```
data
├── IP-28-28-206.h5
├── Indian_pines_gt.mat
├── HU-28-28-150.h5
├── Houston_gt.mat
├── Bw-28-28-151.h5
├── Botswana_gt.mat
└── ...
```


## Usage

Run the following command:

Train on Indian Pines:
python main.py --dataset indian --data_dir ./data --gt_dir ./data



## Acknowledgement

Thanks for EMVCC and SCMVC. We build this library based on the EMVCC's codebase and the SCMVC's codebase.
