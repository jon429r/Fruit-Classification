# Fruit-Classification
This is our solution for the [Fruits Classification](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification) dataset. We employ a CNN for this classification task this repo holds all the code we used to do so.

# Installation 
To install this repo you will need to install it's python depedencies.
```
pip install -r requirements.txt
```

# Data
To get our data you will need to run `fetch_data.py`.
```
python fetch_data.py
```
This will fetch the data from kaggle and adjust the split. The data was previously split at a 97 2 1 ratio but we adjust it to a 80 10 10 split.
Once you have done this the data is ready for training our model.

# Training 
We provide our model and training code in `simple_cnn.py`. The model is defined as the class `SimpleCnn`.

## Usage
You can train the model with various hyperparameters such as epochs, learning rate, and batch size. You can also add additional flags save the model weights and graph the loss & accuracy during training.

```
usage: simple_cnn.py [-h] [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE] [--train] [--save] [--graph]

Fruit Image Classification with Simple CNN

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of training epochs -> Default: 10
  --lr LR               Learning rate -> Default: 0.0001
  --batch_size BATCH_SIZE
                        Batch size for training -> Default: 32
  --train               Enable training mode -> Default: False
  --save                Save the model -> Default: False
  --graph               Plot the graph -> Default: False
```
