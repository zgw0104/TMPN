# TMPN
## Dataset  

We follow the same pre-processing strategy described in the below link.


[https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md](https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md)


You should put datasets into relevant directory.  
* Visual Genome => Datasets/VG  
```bash  
.   
├── Datasets    
│       └── VG    
│           ├── image_data.json   
│           ├── VG-SGG-with-attri.h5
│           ├── VG-SGG-dicts-with-attri.json
│           ├── Category_Type_Info.json
│           └── VG_100k
│                   └── *.png
```  

## Pretrained Faster R-CNN  

We employ the same pretrained Faster R-CNN module corresponding to [BGNN](https://github.com/SHTUPLUS/PySGG)  

* Faster R-CNN
    * [vg](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EQIy64T-EK9Er9y8kVCDaukB79gJwfSsEIbey9g0Xag6lg?e=wkKHJs)  


## Package Install

``` python  

conda create -n tmpn python=3.7.7

conda activate tmpn

conda install -y ipython scipy h5py

pip install ninja yacs cython matplotlib tqdm opencv-python overrides gpustat gitpython ipdb graphviz tensorboardx termcolor scikit-learn==0.23.1

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch

pip install torch-scatter==2.0.7 torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html

pip install torch-sparse -f https://data.pyg.org/whl/torch-1.7.0+cu110.html

pip install torch-geometric

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

cd ..

git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

cd ..

python setup.py build develop

```  

## Implementation  

You should train the **HetSGG** model in **shell/** directory.


* Train  
``` python  
## SGCls
bash shell/tmpn_train_predcls_vg.sh  

```  