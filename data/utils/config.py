class Config:
    def __init__(self,task):
        if "la" in task: # SSL
            self.base_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/Datasets/LASeg/'
            self.save_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/LA_data'
            self.patch_size = (112, 112, 80)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 16
            self.early_stop_patience = 50
            self.batch_size = 2
            self.labeled_num = 4
            
        elif "pancreas" in task: # SSL
            self.base_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/Datasets/Pancreas/'
            self.save_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/Pancreas_data'
            self.patch_size = (96, 96, 96)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 16
            self.early_stop_patience = 50
            self.batch_size = 2
            self.labeled_num = 6
            
        elif "brats2019" in task: 
            self.base_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/Datasets/BraTS2019/'
            self.save_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/Brats2019_data'
            self.patch_size = (96, 96, 96)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 16
            self.early_stop_patience = 50
            self.batch_size = 2
            self.labeled_num = 25
            
        elif "acdc" in task: 
            self.base_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/Datasets/ACDC2017/'
            self.save_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/ACDC2017_data'
            self.patch_size = (256, 256)
            self.num_cls = 4
            self.num_channels = 1
            self.n_filters = 16
            self.early_stop_patience = 50
            self.batch_size = 8
            self.labeled_num = 7
            
        elif "isic2018" in task: # SSL
            self.base_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/Datasets/ISIC2018/'
            self.save_dir = '/d/home/xiuzhen-guo/code/GenericSSL-gxz/ISIC2018_data'
            self.patch_size = (256, 192)
            self.num_cls = 2
            self.num_channels = 3
            self.n_filters = 16
            self.early_stop_patience = 50
            self.batch_size = 2
            self.labeled_num = 259

        elif "synapse" in task: # IBSSL
            self.base_dir = './Datasets/Synapse'
            self.save_dir = './Synapse_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 2

        elif "mmwhs" in task: # UDA
            self.base_dir = './Datasets/MMWHS'
            self.save_dir = './MMWHS_data'
            self.patch_size = (128, 128, 128)
            self.num_cls = 5
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
            self.batch_size = 2

        elif "mnms" in task: # SemiDG
            if "2d" in task:
                self.base_dir = './Datasets/MNMs/Labeled'
                self.save_dir = './MNMS_data_2d'
                self.patch_size = (224, 224)
                self.num_cls = 4
                self.num_channels = 1
                self.n_filters = 32
                self.early_stop_patience = 50
                self.batch_size = 32
            else:
                self.base_dir = './Datasets/MNMs/Labeled'
                self.save_dir = './MNMS_data'
                self.patch_size = (32, 128, 128)
                self.num_cls = 4
                self.num_channels = 1
                self.n_filters = 32
                self.early_stop_patience = 80
                self.batch_size = 4


        else:
            raise NameError("Please provide correct task name, see config.py")


