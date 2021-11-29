import numpy as np 

# Unpickle Function from toronto 
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_files():
    # Unpickling Training Data Batches
    data_batch_1 = unpickle("data/cifar-10-batches-py/data_batch_1")
    data_batch_2 = unpickle("data/cifar-10-batches-py/data_batch_2")
    data_batch_3 = unpickle("data/cifar-10-batches-py/data_batch_3")
    data_batch_4 = unpickle("data/cifar-10-batches-py/data_batch_4")
    data_batch_5 = unpickle("data/cifar-10-batches-py/data_batch_5")

    # Extracting Training Images 
    Training_Images = data_batch_1[b'data']
    Training_Images = np.vstack((Training_Images,data_batch_2[b'data']))
    Training_Images = np.vstack((Training_Images,data_batch_3[b'data']))
    Training_Images = np.vstack((Training_Images,data_batch_4[b'data']))
    Training_Images = np.vstack((Training_Images,data_batch_5[b'data']))

    # Extracting Training Labels
    Training_Labels = data_batch_1[b'labels']
    Training_Labels = np.hstack((Training_Labels,data_batch_2[b'labels']))
    Training_Labels = np.hstack((Training_Labels,data_batch_3[b'labels']))
    Training_Labels = np.hstack((Training_Labels,data_batch_4[b'labels']))
    Training_Labels = np.hstack((Training_Labels,data_batch_5[b'labels']))

    # Extracting Test Data 
    test_batch = unpickle("data/cifar-10-batches-py/test_batch")

    # Extracting Test Images
    Test_Images = test_batch[b'data']

    # Extracting Test Labels
    Test_Labels = np.array(test_batch[b'labels'])

    # Getting Labels List
    batches_meta = unpickle("data/cifar-10-batches-py/batches.meta")

    CIFAR10_LABELS = batches_meta[b'label_names']
    CIFAR10_LABELS = [ x.decode('ascii') for x in CIFAR10_LABELS ]
    # Reshape Images into RGB Channels 

    # Set Image Shape 
    Pixel_shape = 32
    Channel = 3 

    Training_Images = np.reshape(Training_Images, (50000,Channel,Pixel_shape, Pixel_shape))
    Training_Images = Training_Images.transpose(0, 2, 3, 1)

    Test_Images = np.reshape(Test_Images, (10000, Channel,Pixel_shape, Pixel_shape))
    Test_Images = Test_Images.transpose(0, 2, 3, 1)
    
    # Pre-Process Training Images and Test Images 
    Training_Images = Training_Images/255.0

    Test_Images = Test_Images/255.0
    
    print("-------------------------------------------------")
    print("Extraction of CIFAR10 Dataset Done")
    print(f"Shape of Training Images : {Training_Images.shape}")
    print(f"Shape of Training Labels : {Training_Labels.shape}")
    print(f"Shape of Test Images     : {Test_Images.shape}")
    print(f"Shape of Test Labels     : {Test_Labels.shape}")
    print("-------------------------------------------------")

    return ((Training_Images,Training_Labels),(Test_Images,Test_Labels),CIFAR10_LABELS)