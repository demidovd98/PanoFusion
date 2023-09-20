# PanoFusion




### Instructions:

#### Run:

```
# no cuda load

cd '/l/users/20020067/Study/PhD/2 semester/CV-802 (3D vision)/Project/_Ours/PanoFusion/'

source /apps/local/anaconda2023/conda_init.sh
# (local):
conda activate omnifusion_pip
# (shared): 
conda activate /l/users/MODE/envs/omnifusion_pip

python ./OmniFusion/test.py --nrows 4
```


### Other:

#### Create environment:

```
conda activate omnifusion_pip
pip3 install -r requirements.txt
conda install -c conda-forge openexr # not rly?
conda install -c conda-forge openexr-python # yes
pip install mmcv==1.5.0
pip install mmsegmentation==0.23.0
```

#### copy environment to the shared folder:

```
conda create --prefix ./name --clone name
```