# source https://github.com/huitangtang/SRDC-CVPR2020/blob/master/models/model_construct.py
from model.resnet import resnet

def Model_Construct(num_classes):
  arch='resnet50'
  if arch.find('resnet') != -1:
      model = resnet(num_classes)
      return model
  else:
      raise ValueError('The required model does not exist!')
      
