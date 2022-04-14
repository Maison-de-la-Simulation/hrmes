import torch

length = 12
batch_size = 8
epochs = 100
test_length = 10
train_test_split_index = 5000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

msft_dataset_path = "/gpfswork/idris/sos/ssos025/HRMES/MLspinup/IPSLCM6ALR/msftbarot_Omon_IPSL-CM6A-LR_piControl_r1i1p1f1_gn_185001-234912.nc"
mask_dataset_path = "/gpfswork/idris/sos/ssos025/HRMES/MLspinup/INPUTS/eORCA1.2_mesh_mask.nc"

learning_rate = 0.00001
weight_decay = 0.1