# Micrometer: A Foundation Model for Predicting the Mechanical Response of Heterogeneous Materials

Code for [paper]() Micrometer: Micromechanics Transformer for Predicting Mechanical Responses of Heterogeneous Materials.

![Micrometer](./figures/micrometer.png)


## Getting Started

### Installation

Our code has been tested with a Linux environment using the following configuration:

- Python 3.9
- CUDA 12.4
- CUDNN 8.9
- JAX 0.4.26


First, clone the repository:

```angular2html
git clone https://github.com/PredictiveIntelligenceLab/micrometer.git
cd micrometer
```

Then, install the required packages

```angular2html
pip3 install -U pip
pip3 install --upgrade jax jaxlib
pip3 install --upgrade -r requirements.txt
```

Finally, install the package:

```angular2html
pip3 install -e .
```


### Dataset

Our dataset can be downloaded from the following Google Drive links:

| Name                 | Link     |
|----------------------|----------|
| CMME                 | [Link](https://drive.google.com/drive/folders/1eeFrLQkJawuJAcizykwg3kQM1yCyaF74?usp=sharing) |
| Homogenization       | [Link](https://drive.google.com/drive/folders/1nN0LoqwkGVe_k74XIZZNLHvBgADPoC7F?usp=sharing) |
| Multiscale Modelling | [Link](https://drive.google.com/drive/folders/1PBTqFkVn63IfEgz4J3RbvxoUj6ZT5AHe?usp=sharing) |
| Transfer Learning    | [Link](https://drive.google.com/drive/folders/1PIggy_sadd3iX1JSAkIxe5vdbIABszFy?usp=sharing) |



### Training

First, please place the downloaded the dataset and change the data path in
the corresponding config file, e.g. in `configs/base.py`:

```angular2html
dataset.data_path = "path/to/dataset"
```

Then, to train our Micrometer (e.g. `cvit_b_16`), run the following command:

```angular2html
python3 main.py --config=configs/base.py:cvit_b_16
```

The user can also train other models by changing the model name 
in the above command, e.g. `cvit_b_16` to `cvit_l_8`. We also provide UNet or FNO of different configurations as the backbone model,
which can be found in `configs/models.py`.



To specify the GPU device, for example, use the following command:

```angular2html
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --config=configs/base.py:cvit_b_16
```

### Evaluation

To evaluate the trained model (e.g. `cvit_b_16`), run the following command:

```angular2html
python3 main.py --config=configs/base.py:cvit_b_16 --config.mode=eval
```

### Homogenization

With the pre-trained model, we can perform homogenization by running the following command:

```angular2html
python3 main.py --config=configs/homo.py:cvit_b_16
```

### Mutli-scale Modeling


With the pre-trained model, we can perform multiscale modelling

```angular2html
python3 main.py --config=configs/multiscale.py:cvit_b_16
```


### Transfer Learning

We can also fine-tune our pretrained model on new datasets, which is configured in
`finetune_vf.py ` or `finetune_ch.py`

```angular2html
python3 main.py --config=configs/finetune_ch.py:cvit_b_16 
```


## Checkpoints







## Citation






