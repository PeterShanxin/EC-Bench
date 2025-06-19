cd CLEAN/app/
conda create -n clean python==3.10.4 -y
conda activate clean
pip install -r requirements.txt
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch (GPU)
python3 build.py install
cd ../
git clone https://github.com/facebookresearch/esm.git
mkdir data/esm_data