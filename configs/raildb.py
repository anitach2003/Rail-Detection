# DATA
dataset = 'raildb'
data_root = '/kaggle/working/Rail-Detection/Rail-DB/'

# TRAIN
epoch = 50
batch_size = 64
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = 'vit_b_16'
griding_num = 200
cls_num_per_lane = 52

# EXP
note = 'test'

log_path = '/kaggle/working/Rail-Detection/Rail-DB/utils'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = '/kaggle/working/Rail-Detection/Rail-DB/log/test_model.pth'
test_work_dir = '/kaggle/working/Rail-Detection/Rail-DB/test/'

num_lanes = 4
type = 'all'
