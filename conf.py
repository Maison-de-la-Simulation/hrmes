import torch

length = 12
batch_size = 8
epochs = 100
test_length = 10
train_test_split_ratio = 0.8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mask_dataset_path = "/gpfswork/idris/sos/ssos025/HRMES/MLspinup/INPUTS/eORCA1.2_mesh_mask.nc"
#msft_dataset_path = "/gpfswork/idris/sos/ssos025/HRMES/MLspinup/IPSLCM6ALR/msftbarot_Omon_IPSL-CM6A-LR_piControl_r1i1p1f1_gn_185001-234912.nc"

msft_dataset_path_prefix = "/gpfswork/rech/omr/romr004/MLspinup/CM65v406/MSFT"
msft_dataset_path = [[
    'CM65v406-LR-pi-06_18600101_18691231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_18700101_18791231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_18800101_18891231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_18900101_18991231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19000101_19091231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19100101_19191231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19200101_19291231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19300101_19391231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19400101_19491231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19500101_19591231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19600101_19691231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19700101_19791231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19800101_19891231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_19900101_19991231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20000101_20091231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20100101_20191231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20200101_20291231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20300101_20391231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20400101_20491231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20500101_20591231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20600101_20691231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20700101_20791231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20800101_20891231_1M_MSFT.nc',
    'CM65v406-LR-pi-06_20900101_20991231_1M_MSFT.nc',],
    ['CM65v406-LR-pi-NWGHT-02_19000101_19091231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_19100101_19191231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_19200101_19291231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_19300101_19391231_1M_MSFT.nc', 
    'CM65v406-LR-pi-NWGHT-02_19400101_19491231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_19500101_19591231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_19600101_19691231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_19700101_19791231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_19800101_19891231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_19900101_19991231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20000101_20091231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20100101_20191231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20200101_20291231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20300101_20391231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20400101_20491231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20500101_20591231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20600101_20691231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20700101_20791231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20800101_20891231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_20900101_20991231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21000101_21091231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21100101_21191231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21200101_21291231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21300101_21391231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21400101_21491231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21500101_21591231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21600101_21691231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21700101_21791231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21800101_21891231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_21900101_21991231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22000101_22091231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22100101_22191231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22200101_22291231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22300101_22391231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22400101_22491231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22500101_22591231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22600101_22691231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22700101_22791231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22800101_22891231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_22900101_22991231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23000101_23091231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23100101_23191231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23200101_23291231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23300101_23391231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23400101_23491231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23500101_23591231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23600101_23691231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23700101_23791231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23800101_23891231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_23900101_23991231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24000101_24091231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24100101_24191231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24200101_24291231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24300101_24391231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24400101_24491231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24500101_24591231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24600101_24691231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24700101_24791231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24800101_24891231_1M_MSFT.nc',
    'CM65v406-LR-pi-NWGHT-02_24900101_24991231_1M_MSFT.nc',],
    ['CM65v6-LR-pi-JZ-03_18600101_18691231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_18700101_18791231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_18800101_18891231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_18900101_18991231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19000101_19091231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19100101_19191231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19200101_19291231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19300101_19391231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19400101_19491231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19500101_19591231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19600101_19691231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19700101_19791231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19800101_19891231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_19900101_19991231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20000101_20091231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20100101_20191231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20200101_20291231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20300101_20391231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20400101_20491231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20500101_20591231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20600101_20691231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20700101_20791231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20800101_20891231_1M_MSFT.nc',
    'CM65v6-LR-pi-JZ-03_20900101_20991231_1M_MSFT.nc']
]

learning_rate = 0.00001
weight_decay = 0.1