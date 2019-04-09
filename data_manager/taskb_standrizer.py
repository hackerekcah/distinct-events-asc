import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


class TaskbStandarizer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.scaler_h5 = os.path.dirname(data_manager.dev_h5_path) + '/TaskbDevMatrixScaler.h5'

    def calc_mu_sigma(self, data):
        """
        calc mu and sigma give data numpy
        :param data: (batch, frequency, time)
        :return: mu, sigma numpy vector
        """
        # transpose to (batch, time, frequency)
        data = np.transpose(data, [0, 2, 1])
        # mean, std
        mu = np.mean(data, axis=(0, 1))
        sigma = np.std(data, axis=(0, 1))

        return mu, sigma

    def mu_sigma_by_device(self, mode='train', device='a'):
        """
        given mode and device, return mu and sigma
        :param mode:
        :param device:
        :return:
        """

        data, _ = self.data_manager.load_dev(mode=mode, devices=device)
        return self.calc_mu_sigma(data)

    def create_scaler_h5(self):
        """
        create scaler h5 file, data is f[mode][device]['mu'] and f[mode][device]['sigma']
        :return:
        """
        if os.path.exists(self.scaler_h5):
            print("[LOGGING]: " + self.scaler_h5 + " exists!")
            return
        with h5py.File(self.scaler_h5, 'w') as f:
            for mode in ['train', 'test']:
                for device in ['a', 'b', 'c', 'p', 'A']:
                    grp = f.create_group(mode + '/' + device)
                    grp['mu'], grp['sigma'] = self.mu_sigma_by_device(mode=mode, device=device)
        f.close()

    def load_scaler(self, mode='train', device='a'):
        """
        load mu and sigma given mode and device
        :param mode:
        :param device:
        :return:
        """
        with h5py.File(self.scaler_h5, 'r') as f:
            mu = f[mode][device]['mu'].value
            sigma = f[mode][device]['sigma'].value
            return mu, sigma

    def load_dev_standrized(self, mode='train', device='a', norm_device='a'):
        """
        given mode and device, scale using train scaler , and return data and label
        :param mode:
        :param device:
        :param norm_device: use scaler from norm device
        :return:
        """
        data, label = self.data_manager.load_dev(mode=mode, devices=device)
        # Take Care!!! only scale using train mu and sigma
        mu, sigma = self.load_scaler(mode='train', device=norm_device)

        data = np.transpose(data, [0, 2, 1])
        data = (data - mu) / sigma
        # transpose back to (batch, frequency, time)
        data = np.transpose(data, [0, 2, 1])

        print("[LOGGING] Normalize using device {}".format(norm_device))

        return data, label

    def load_dev_fname_standrized(self, mode='train', device='a', norm_device='a'):
        """
        given mode and device, scale using train scaler , and return data and label
        :param mode:
        :param device:
        :param norm_device: use scaler from norm device
        :return:
        """
        data, label, fnames = self.data_manager.load_dev_with_fnames(mode=mode, devices=device)
        # Take Care!!! only scale using train mu and sigma
        mu, sigma = self.load_scaler(mode='train', device=norm_device)

        data = np.transpose(data, [0, 2, 1])
        data = (data - mu) / sigma
        # transpose back to (batch, frequency, time)
        data = np.transpose(data, [0, 2, 1])

        print("[LOGGING] Normalize using device {}".format(norm_device))

        return data, label, fnames

    def load_normed_spec_by_name(self, wav_name=None, norm_device=None):
        spec_data = self.data_manager.load_spec_by_name(wav_name)
        # Take Care!!! only scale using train mu and sigma
        if norm_device:
            mu, sigma = self.load_scaler(mode='train', device=norm_device)
        else:
            mu, sigma = self.load_scaler(mode='train', device=wav_name[-5])

        spec_data = np.transpose(spec_data, [0, 2, 1])
        spec_data = (spec_data - mu) / sigma
        # transpose back to (batch, frequency, time)
        spec_data = np.transpose(spec_data, [0, 2, 1])

        return spec_data

    def plot_scaler(self, save_path=None):
        """
        plot mu and sigma for train/val/device, save fig if save_path specified, otherwise plot.
        :param save_path:
        :return:
        """
        if save_path and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        device_list = ['a', 'b', 'c']
        # device_list = ['a', 'b', 'c', 'p']

        plt.figure()
        plt.subplot(2, 2, 1)
        for device in device_list:
            mu, _ = self.load_scaler(mode='train', device=device)
            plt.plot(np.arange(40), mu)
        plt.legend(['train_a', 'train_b', 'train_c', 'train_p'])
        plt.title("train/mu")

        plt.subplot(2, 2, 2)
        for device in device_list:
            mu, _ = self.load_scaler(mode='test', device=device)
            plt.plot(np.arange(40), mu)
        plt.legend(['val_a', 'val_b', 'val_c', 'val_p'])
        plt.title("val/mu")

        plt.subplot(2, 2, 3)

        for device in device_list:
            _, sigma = self.load_scaler(mode='train', device=device)
            plt.plot(np.arange(40), sigma)
        plt.legend(['train_a', 'train_b', 'train_c', 'train_p'])
        plt.title("train/sigma")

        plt.subplot(2, 2, 4)

        for device in device_list:
            _, sigma = self.load_scaler(mode='test', device=device)
            plt.plot(np.arange(40), sigma)
        plt.legend(['val_a', 'val_b', 'val_c', 'val_p'])
        plt.title("val/sigma")

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=500)
        else:
            plt.show()


if __name__ == '__main__':
    from data_manager.dcase18_taskb import Dcase18TaskbData
    ori_data_manager = Dcase18TaskbData()
    ori_standarizer = TaskbStandarizer(data_manager=ori_data_manager)
    ori_standarizer.create_scaler_h5()
    # ori_fig_path = '../data_manager/scaler_plots/ori_scaler.png'
    # ori_standarizer.plot_scaler(save_path=ori_fig_path)

