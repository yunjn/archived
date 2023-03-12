# 公路浓雾预警

## introduction

* Based on `PyTorch` and  `ResNet`  identify the level of dense highway fog (dense_fog ,mist,no_fog)
* Provide local applications ~~and cloud services (TensorFlow Servering)~~

![demo](./resources/images/demo.gif)

## development environment

* [miniconda3](https://docs.conda.io/en/latest/miniconda.html)

* [PyTorch](pytorch.org)

## usage

### GUI

* `select:` Process the selected image or video 
* `NET:`[IP摄像头](https://summer907.lanzous.com/ich68fi)，Enter the lan ip
* `clean:`Clean up

### train

####  use trained

`python GUI.py`

#### retrain

```txt
└─fog_data
    └─fog_img
       ├─dense_fog
       ├─mist
       └─no_fog
```

`python split.py`

```
└─fog_data
    ├─fog_img
    │  ├─dense_fog
    │  ├─mist
    │  └─no_fog
    ├─val
    │  ├─dense_fog
    │  ├─mist
    │  └─no_fog
    └─train
        ├─dense_fog
        ├─mist
        └─no_fog
```

* `python train.py`

  > Choose thee appropriate batch_size value

## update

* 2021/03/20 改用`PyTorch`; 调整代码结构，增加复用，`online`模块暂时舍弃
