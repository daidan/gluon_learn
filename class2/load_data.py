# -*- coding: UTF-8 -*-

'''
Created on 2018年1月4日

@author: daidan
'''
from mxnet import gluon


def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

def get_MNIST():
    mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
    
    return mnist_train, mnist_test

def get_FashionMNIST():
    fashion_mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
    fashion_mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
    
    return fashion_mnist_train, fashion_mnist_test

def get_CIFAR10():
    cifar10_train = gluon.data.vision.CIFAR10(train=True, transform=transform)
    cifar10_test = gluon.data.vision.CIFAR10(train=False, transform=transform)
    
    return cifar10_train, cifar10_test

def get_CIFAR100():
    cifar100_train = gluon.data.vision.CIFAR100(train=True, transform=transform)
    cifar100_test = gluon.data.vision.CIFAR100(train=False, transform=transform)
    
    return cifar100_train, cifar100_test
    

def get_dataloader(get_data=get_FashionMNIST, batch_size=256):
        
    data_train, data_test = get_data()
        
    batch_dataloader_train = gluon.data.DataLoader(data_train, batch_size, shuffle=True)
    batch_dataloader_test = gluon.data.DataLoader(data_test, batch_size, shuffle=False)
    
    return batch_dataloader_train, batch_dataloader_test


#------------------------------------------------------------------------------ 

def show_plot(get_data=get_FashionMNIST):
    
    data_train, data_test = get_data()
    
    import matplotlib.pyplot as plt

    def show_images(images):
        n = images.shape[0]
        _, figs = plt.subplots(1, n, figsize=(15, 15))
        for i in range(n):
            figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
            figs[i].axes.get_xaxis().set_visible(False)
            figs[i].axes.get_yaxis().set_visible(False)
        plt.show()

    def get_text_labels(label):
        text_labels = [
            't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        return [text_labels[int(i)] for i in label]

    data, label = data_train[0:9]
    show_images(data)
    print(get_text_labels(label))
show_plot(get_data=get_MNIST)