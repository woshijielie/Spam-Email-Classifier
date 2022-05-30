#!/bin/bash
sudo -u ec2-user -i
set -e

NOTEBOOK_FILE="/home/ec2-user/SageMaker/smlambdaworkshop/training/sms_spam_classifier_mxnet.ipynb"

echo '{                                           
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "mxnet",
    "image_data_format": "channels_last"
}' | sudo tee -a /root/.keras/keras_mxnet.json

sudo mkdir /root/.dl_binaries
sudo mkdir /root/.dl_binaries/mxnet
sudo mkdir /root/.dl_binaries/mxnet/cpu_2.7_3.6

echo "upgrade sagamaker"
echo "activate mxnet_p36"
source /home/ec2-user/anaconda3/bin/activate mxnet_p36 && pip install mxnet && pip install --upgrade sagemaker
echo "execute jupyter notebook"
jupyter trust "$NOTEBOOK_FILE"
nohup jupyter nbconvert --to notebook --inplace --execute "$NOTEBOOK_FILE" --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1&
echo "finish executing jupyter notebook"

# source /home/ec2-user/anaconda3/bin/deactivate