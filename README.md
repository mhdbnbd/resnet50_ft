## resnet50_ft ##
Finetuning Resnet50 and benchmarking it on transfer tasks within Office31 and between SVHN and MNIST

## Requirements ##

matplotlib==3.2.2  
numpy==1.19.5  
opencv_contrib_python==4.1.2.30  
scipy==1.4.1  
requests==2.23.0  
torchvision==0.8.2+cu101  
torch==1.7.1+cu101

## Run code ##


1. load model

```python
import torch
from model.model_construct import Model_Construct
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Model_Construct(num_classes).to(device)
  """
  num_classes: the number of classes in the dataset (e.g. num_classes=31 for "Office31" and num_classes=10 for "MNIST/SVHN")
  """
```

2. load data

```python
from data.preprocess import load_ds

load_ds(datasets, batch_size, num_workers, grayscale, normalize, split_path, ds_path)

  """
  datasets: either 'office31' or 'mnist/svhn'
  batch_size: batch size for dataloaders
  num_workers: number of workers for dataloaders
  grayscale: if True applies a transform that converts grayscaled images (1-channel) to RGB images (3-channels) (recommended for 'mnist/svhn')
  normalize: if True normalizes images with natural images mean,std
  split_path: if datasets="office31" specify the path to split_structure folder
  ds_path: if datasets="office31" specify the path where to build office31 splits
  Returns:
  -----------------
  n dataloaders (n=9 for 'office31' corresponding resp. to amazon, webcam, dslr * full(target test), half_1(source train), half_2(source validation)
                 n=4 for 'mnist/svhn' corresponding resp. to mnist, svhn * train,val/test)
  Raises:
  -----------------
  ValueError: If neither datasets='office31' or datasets='mnist/svhn'
  """
  ```

3. define training data as dictionary

```python
source_dataloader = {'train': amazon_halfloader, 'val': amazon_half2loader}
```

4. training and testing

```python
from train import train_model, predict_batchwise

train_model(model, source_dataloader, num_epochs, lr,  weight_decay, instNorm)
      """
      model: model to train
      dataloaders: dictionary of 'train'/'val' dataloaders
      num_epochs: number of epochs
      lr: learning rate for the optimizer Adagrad
      lr_decay: learning rate decay for the optimizer Adagrad
      weight_decay: weight decay for the optimizer Adagrad
      instNorm: if True applies torch.nn.InstanceNorm2d to input instances (recommended for 'mnist/svhn')

      Returns:
      -----------------
      trained model and validation accuracies history
      """

predict_batchwise(model, dataloader)

      """
      model: trained model
      dataloader: target test dataloader

      Returns:
      -----------------
      predictions accuracy
      """
   ```
    
# Example # 

Assuming the code is cloned on "/content", the following benchmarks the model on Office31 (Amazon(A) to Webcam(W) and DSLR(D))

```python
print("----Office31----")
# load model
model_31 = Model_Construct(num_classes=31).to(device)
# load dataloaders
amazon_loader, amazon_halfloader, amazon_half2loader, webcam_loader, webcam_halfloader, webcam_half2loader, dslr_loader, dslr_halfloader, dslr_half2loader = load_ds(datasets="office31", batch_size=64, num_workers=4, grayscale=False, normalize=False, split_path=os.path.join("/content", "data/splits_structure"), ds_path=os.path.join("/content", "data/Office31"))
print("--Source: Amazon--")
# define training data
source_dataloader = {'train': amazon_halfloader, 'val': amazon_half2loader}
# train
amazon_model, amazon_hist = train_model(model_31, source_dataloader, num_epochs=200, lr=0.001,  weight_decay=0.0001, instNorm=False)
# test
print("--Target: Webcam, DSLR--")
print("Prediction on webcam:")
predict_batchwise(amazon_model, webcam_loader)
print("Prediction on DSLR:")
predict_batchwise(amazon_model, dslr_loader)
```
The following benchmarks the model on SVHN/MNIST (MNIST to SVHN)

```python
#load model
model_10 = Model_Construct(num_classes=10).to(device)
#load dataloaders
mnist_trainloader, mnist_testloader, svhn_trainloader, svhn_testloader = load_ds(datasets="mnist/svhn", batch_size=64, num_workers=4, grayscale = True, normalize = False)
print("--Source: MNIST--")
#train
source_dataloader = {'train': mnist_trainloader, 'val': mnist_testloader}
mnist_model, mnist_hist = train_model(model_10, source_dataloader, num_epochs=40, lr=0.00025, weight_decay=0, instNorm=True)
#test
print("--Target: SVHN--")
print("-Prediction on SVHN:")
predict_batchwise(mnist_model, svhn_testloader)
```

# Performance #

Running the parameters from the examples above (e.g. num_epochs, lr ...), the following results are returned :

A &#8594; W      | D &#8594; W      | W &#8594; D      | A &#8594; D      | D &#8594; A      | W &#8594; A      | Avg
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
~76.9         | 90.4 (±1.0)  | ~96.80 (±1.0)   | ~81.7         | 53.3 (±1.8)   | 57.0 (±1.0)     | ~76.0

SVHN &#8594; MNIST      | MNIST &#8594; SVHN     | Avg
-------------        | -------------       | ------------- 
~73.29                 | ~20.0                  | ~46.65      

