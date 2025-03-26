from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, input_tensor, output_tensor, window_size):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.window_size = window_size

    def __len__(self):
        return len(self.input_tensor) - self.window_size + 1

    def __getitem__(self, idx):
        input_samples = self.input_tensor[idx:idx+self.window_size]
        output_sample = self.output_tensor[idx:idx+self.window_size]

        return input_samples, output_sample


class My_eval_trans_Dataset(Dataset):
    def __init__(self, x_imu, pose, tran,window_size):
        self.x_imu = x_imu
        self.pose = pose
        self.tran = tran
        self.window_size = window_size

    def __len__(self):
        return len(self.x_imu) - self.window_size + 1

    def __getitem__(self, idx):
        x_imu_samples = self.x_imu[idx:idx+self.window_size]
        pose_sample = self.pose[idx:idx+self.window_size]
        tran_sample = self.tran[idx:idx+self.window_size]
        return x_imu_samples, pose_sample,tran_sample


class My_Translation_Dataset(Dataset):
    def __init__(self, input_tensor1, input_tensor2,output_tensor, window_size):
        self.input_tensor1 = input_tensor1
        self.input_tensor2 = input_tensor2
        self.output_tensor = output_tensor
        self.window_size = window_size

    def __len__(self):
        return len(self.input_tensor1) - self.window_size + 1

    def __getitem__(self, idx):
        input_samples1 = self.input_tensor1[idx:idx+self.window_size]
        input_samples2 = self.input_tensor2[idx:idx+self.window_size]
        output_sample = self.output_tensor[idx+self.window_size-1:idx+self.window_size]

        return input_samples1, input_samples2,output_sample


