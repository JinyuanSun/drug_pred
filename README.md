# drug_pred

### Install
Clone this repo and install
```
git clone https://github.com/JinyuanSun/drug_pred.git
cd drug_pred
pip install -r requirements.txt
```
### Usage:
#### Quickstart using demo data.
```bash
# train your model using a demo data
cp demo/D1D2_107.fas ./
cp demo/drug_data.txt ./
python3 drug_predictor.py D1D2_107.fas train --drug_data drug_data.txt --model_name d1d2_FZ
```
Model weights saved in `d1d2_FZ.pth`.
#### Run inference:
```bash
# run inference (aka prediction) using demo data
python3 drug_predictor.py D1D2_107.fas inference --model_name d1d2_FZ
```
Output in `D1D2_107.fas.dscore`.