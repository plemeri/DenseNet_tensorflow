# DenseNet_tensorflow
tensorflow implementation of Densely Connected Convolutional Networks

## Requirements
* Tensorflow 1.x - GPU version recommended
* Python 3.x


## Dataset

Please download dataset from this [link](https://drive.google.com/drive/folders/1kr0bGAmf3xuOUkw1DTA8gSBsO9LTObyk?usp=sharing)
Both Cifar10 and MNIST dataset are converted into tfrecords format for conveinence. Put `train.tfrecords`, `test.tfrecords` files into `dataset/cifar10`, `dataset/mnist`

You can create tfrecord file with your own dataset with `dataset/dataset_generator.py`.
```sh
python dataset_generator.py --image_dir ./cifar10/test/images --label_dir ./cifar10/test/labels --output_dir ./cifar10 --output_filename test.tfrecord
```

Options:

- `--image_dir` (str) - directory of your image files. it is recommended to set the name of images to integers like `0.png`
- `--label_dir` (str) - directory of your label files. it is recommended to set the name of images to integers like `0.txt`. label text file must contain class label in integer like `8`. 
- `--output_dir` (str) - directory for output tfrecord file.
- `--outpuf_filename` (str) - filename of output tfrecord file.

## Training

**Cifar 10**
```sh
python train.py --class_num 10 --image_shape 32 32 3 --blocks 3 --layers 12 12 12 growth_rate 12 --dropout_rate 0.2 --compression_factor 1.0 --init_subsample False --learning_rate 0.1 --label_smoothing 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 100 --epochs 60 --checkpoint_dir ./checkpoint --checkpoint_name densenet_cifar10 --train_record_dir ./dataset/cifar10/train.tfrecord --val_record_dir ./dataset/cifar10/test.tfrecord
```

Options:
- `--class_num` (int) - output number of class. Cifar10 has 10 classes.
- `--image_shape` (int nargs) - shape of input image. Cifar10 has 32 32 3 shape.
- `--blocks` (int) - the number of dense blocks
- `--layers` (int nargs) - the number of layers for each block. you need to provide them for each block
- `--growth_rate` (int) - growth rate of densenet
- `--dropout_rate` (float) - dropout rate
- `--compression_factor` (float) - compression factor for transition layer. 1.0 for no compressing.
- `--init_subsample` (bool) - do subsampling (striding) if true
- `--learning_rate` (float) - initial learning rate
- `--label_smoothing` (float) - label smoothing factor
- `--momentum` (float) - momentum from momentum optimizer
- `--weight_decay` (float) - weight decay factor
- `--train_set_size` (int) - number of training data. Cifar10 has 50000 data.
- `--val_set_size` (int) - number of validating data. I used test data for validation, so there are 10000 data.
- `--batch_size` (int) - size of mini batch
- `--epochs` (int) - number of epoch
- `--checkpoint_dir` (str) - directory to save checkpoint
- `--checkpoint_name` (str) - file name of checkpoint
- `--train_record_dir` (str) - file location of training set tfrecord
- `--test_record_dir` (str) - file location of test set tfrecord (for validation)

**MNIST**
```sh
python train.py --class_num 10 --image_shape 28 28 1 --blocks 3 --layers 12 12 12 growth_rate 12 --dropout_rate 0.0 --compression_factor 1.0 --init_subsample False --learning_rate 0.1 --label_smoothing 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 100 --epochs 60 --checkpoint_dir ./checkpoint --checkpoint_name densenet_mnist --train_record_dir ./dataset/mnist/train.tfrecord --val_record_dir ./dataset/mnist/test.tfrecord
```

Options:
- options are same as Cifar10

**Cifar100**
(19.04.16 added)
```sh
python train.py --class_num 100 --image_shape 32 32 3 --blocks 3 --layers 12 12 12 growth_rate 12 --dropout_rate 0.0 --compression_factor 1.0 --init_subsample False --learning_rate 0.1 --label_smoothing 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 100 --epochs 60 --checkpoint_dir ./checkpoint --checkpoint_name densenet_cifar100 --train_record_dir ./dataset/cifar100/train.tfrecord --val_record_dir ./dataset/cifar100/test.tfrecord
```

Options:
- options are same as Cifar10

## Testing
```sh
python test.py --class_num 10 --checkpoint_dir ./checkpoint/best --test_record_dir ./dataset/cifar10/test.tfrecord --batch_size 256
```
Options:
- `--class_num` (int) - the number of classes
- `--checkpoint_dir` (str) - directory for the checkpoint you want to load and test
- `--test_record_dir` (str) - directory for the test dataset
- `--batch_size` (int) - batch size for testing

test.py loads network graph and tensors from meta data and evalutes.
