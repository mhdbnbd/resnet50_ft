import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy


from data.office31_build import build_dataset


def load_ds(datasets="office31", batch_size=64, num_workers=4, grayscale=False, normalize=False, split_path=os.path.join("data", "splits_structure"),
            ds_path=os.path.join("Office31")):
    """
  download dataset, apply transforms and build loaders accordingly
  Parameters:
  -----------------
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

    transform_train_ls = [transforms.Resize(256),
                          transforms.RandomCrop(224),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor()]

    transform_test_ls = [transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor()]

    if normalize == True:
        [ls.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])) for ls in
         [transform_train_ls, transform_test_ls]]

    if grayscale == True:
        [ls.insert(0, transforms.Grayscale(num_output_channels=3)) for ls in [transform_train_ls, transform_test_ls]]

    if datasets == "office31":
        [transform_train_ls.append(lam) for lam in [transforms.Lambda(lambda x: _random_affine_augmentation(x)),
                                                    transforms.Lambda(lambda x: _gaussian_blur(x, 0.1))]]

    transform_train = transforms.Compose(transform_train_ls)
    transform_test = transforms.Compose(transform_test_ls)

    if datasets == "office31":

        # requires split_structure
        build_dataset(ds_path=ds_path, split_path=split_path)

        amazon = torchvision.datasets.ImageFolder(os.path.join(ds_path, "amazon/images"), transform_test)
        amazon_loader = torch.utils.data.DataLoader(amazon, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                    pin_memory=True, drop_last=True)

        amazon_half = torchvision.datasets.ImageFolder(os.path.join(ds_path, "amazon_half/images"), transform_train)
        amazon_halfloader = torch.utils.data.DataLoader(amazon_half, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers, pin_memory=True)

        amazon_half2 = torchvision.datasets.ImageFolder(os.path.join(ds_path, "amazon_half2/images"), transform_test)
        amazon_half2loader = torch.utils.data.DataLoader(amazon_half2, batch_size=batch_size, shuffle=False,
                                                         num_workers=num_workers, pin_memory=True)

        webcam = torchvision.datasets.ImageFolder(os.path.join(ds_path, "webcam/images"), transform_test)
        webcam_loader = torch.utils.data.DataLoader(webcam, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                    pin_memory=True)

        webcam_half = torchvision.datasets.ImageFolder(os.path.join(ds_path, "webcam_half/images"), transform_train)
        webcam_halfloader = torch.utils.data.DataLoader(webcam_half, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers, pin_memory=True)

        webcam_half2 = torchvision.datasets.ImageFolder(os.path.join(ds_path, "webcam_half2/images"), transform_test)
        webcam_half2loader = torch.utils.data.DataLoader(webcam_half2, batch_size=batch_size, shuffle=False,
                                                         num_workers=num_workers, pin_memory=True)

        dslr = torchvision.datasets.ImageFolder(os.path.join(ds_path, "dslr/images"), transform_test)
        dslr_loader = torch.utils.data.DataLoader(dslr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  pin_memory=True)

        dslr_half = torchvision.datasets.ImageFolder(os.path.join(ds_path, "dslr_half/images"), transform_train)
        dslr_halfloader = torch.utils.data.DataLoader(dslr_half, batch_size=batch_size, shuffle=True,
                                                      num_workers=num_workers, pin_memory=True)

        dslr_half2 = torchvision.datasets.ImageFolder(os.path.join(ds_path, "dslr_half2/images"), transform_test)
        dslr_half2loader = torch.utils.data.DataLoader(dslr_half2, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_workers, pin_memory=True)

        print(
            "9 dataloaders returned: amazon, webcam, dslr * full(target test), half_1(source train), half_2(source validation)")
        return amazon_loader, amazon_halfloader, amazon_half2loader, webcam_loader, webcam_halfloader, webcam_half2loader, dslr_loader, dslr_halfloader, dslr_half2loader

    elif datasets == "mnist/svhn":

        # hotfix for mnist download error
        # spurce https://github.com/pytorch/vision/issues/1938
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        mnist_trainset = torchvision.datasets.MNIST("MNIST/processed/training.pt", train=True, transform=transform_train,
                                                    download=True)
        mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers, pin_memory=True)

        mnist_testset = torchvision.datasets.MNIST("MNIST/processed/test.pt", train=False, transform=transform_test,
                                                   download=True)
        mnist_testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_workers, pin_memory=True)

        svhn_trainset = torchvision.datasets.SVHN("SVHN", split='train', transform=transform_train, download=True)
        svhn_trainloader = torch.utils.data.DataLoader(svhn_trainset, batch_size=batch_size, shuffle=True,
                                                       num_workers=num_workers, pin_memory=True)

        svhn_testset = torchvision.datasets.SVHN("SVHN", split='test', transform=transform_test, download=True)
        svhn_testloader = torch.utils.data.DataLoader(svhn_testset, batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers, pin_memory=True)

        print("4 dataloaders returned: mnist,svhn * train, val/test")

        return mnist_trainloader, mnist_testloader, svhn_trainloader, svhn_testloader

    # elif:
    #   define new_trainset/new_testset
    #   define new_trainloader/new_testloader
    #   return new_trainloader/new_testloader

    else:
        raise ValueError('please choose a dataset "office31" "mnist/svhn"')


def _random_affine_augmentation(x):
    """
a transform applied to 'office31' on training sets
(From https://github.com/huitangtang/SRDC-CVPR2020/blob/master/data/prepare_data.py)
    """


    M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0],
                    [np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
    rows, cols = x.shape[1:3]
    dst = cv2.warpAffine(np.transpose(x.numpy(), [1, 2, 0]), M, (cols, rows))
    dst = np.transpose(dst, [2, 0, 1])
    return torch.from_numpy(dst)


def _gaussian_blur(x, sigma=0.1):
    """
     a transform applied to 'office31' on training sets
     (From https://github.com/huitangtang/SRDC-CVPR2020/blob/master/data/prepare_data.py)
      """
    ksize = int(sigma + 0.5) * 8 + 1
    dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
    return torch.from_numpy(dst)
