The official implementation for "GCLFormer: Unbiased Augmentation with GraphTransformer for Graph Contrastive Learning"

### How to run our codes?

1. Install the required package according to `requirements.txt`
2. Create a folder `datasets` and download the corresponding dataset
3. Simplely run the program as follows:
```shell
#node_classification
python train.py --dataset Cora --task node_classification
#link_prediction
python train.py --dataset Cora --task link_prediction
#clustering
python train.py --dataset Cora --task clustering
```