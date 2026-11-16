from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.add_model.HybridDataCritic import HybridDataCritic
from models.add_model.PurifierDiffusion import PurifierDiffusion
from models.model import WPMixer
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
    def _build_model(self):
        model_dict = {'WPMixer': WPMixer}
        model = model_dict[self.args.model](self.args.enc_in,
                                            self.args.c_out,
                                            self.args.seq_len,
                                            self.args.pred_len,
                                            self.args.d_model,
                                            self.args.dropout,
                                            self.args.embedding_dropout,
                                            self.device,
                                            self.args.batch_size,
                                            self.args.tfactor,
                                            self.args.dfactor,
                                            self.args.wavelet,
                                            self.args.level,
                                            self.args.patch_len,
                                            self.args.stride,
                                            self.args.no_decomposition,
                                            self.args.use_amp,
                                            self.args.use_kan,
                                            self.args.kan_order,
                                            self.args.prototype_num,
                                            self.args.diffusion_step
                                        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion
    def _select_criterion(self):
        criterion = {'mse': torch.nn.MSELoss(), 'smoothL1': torch.nn.SmoothL1Loss()}
        try:
            return criterion[self.args.loss]
        except KeyError as e:
            raise ValueError(f"Invalid argument: {e} (loss: {self.args.loss})")

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                outputs, att_weights, xA, xD = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x[:, :, f_dim:].detach()

                recon_loss = criterion(pred, true)
                # 3. 计算记忆库熵损失 (Entropy Loss)
                # att_weights 形状通常为 [Batch*Patch, N_items]
                # 目标是让分布趋向于 one-hot，即只有少数条目权重高
                entropy_loss = torch.mean(-att_weights * torch.log(att_weights + 1e-12))
                loss = recon_loss + 0.1 * entropy_loss

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        #获取训练集测试集的数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 设置模型的目录保存文件
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)  # number of batches
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # --- 核心新增：初始化评判器和按样本计算的 Loss ---
        critic = HybridDataCritic(warmup_epochs=5, sigma_factor=1.5, device=self.device)
        # 获取基础 Loss 的类（MSE 或 SmoothL1），重新实例化为 reduction='none' 保留每个样本自己的 loss 值，输出一整串（batch 里每个样本一个 loss）。
        criterion_none = type(criterion)(reduction='none')

        f_dim_len = self.args.c_out if self.args.features != 'MS' else 1

        purifier = PurifierDiffusion(seq_len=self.args.seq_len, channels=f_dim_len, timesteps=self.args.diffusion_step).to(self.device)
        # 给 purifier 单独配一个优化器
        purifier_optim = optim.Adam(purifier.parameters(), lr=1e-5)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            purifier.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                purifier_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                # outputs = self.model(batch_x)
                outputs, att_weights, xA, xD = self.model(batch_x)
                # --- 1. 计算纯净度权重 (Soft-Mask) ---
                pure_weights = critic.get_weights(xA, xD, att_weights, epoch)  # 形状 [B]

                # MS是根据多变量检测单变量，所以当为MS只在最后一个变量计算损失；当为M或S时，切片等价于不切，此时是单变量检测单变量或者多变量检测多变量
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x_sliced = batch_x[:, :, f_dim:]
                # ==========================================================
                # target 默认是原始数据
                batch_x_target = batch_x_sliced.clone()
                diff_loss = torch.tensor(0.0).to(self.device)

                # --- [创新点 2：动态免疫与提纯逻辑] ---
                # 设定阈值：>0.7 认为是干净的，<0.3 认为是脏的
                clean_mask = pure_weights > 0.7
                dirty_mask = pure_weights < 0.3

                if epoch >= critic.warmup_epochs:
                    # 用干净数据训练净化器
                    if clean_mask.sum() > 0:
                        clean_samples = batch_x_sliced[clean_mask]
                        diff_loss = purifier.get_loss(clean_samples)

                    # 洗白脏数据
                    if dirty_mask.sum() > 0:
                        purifier.eval()  # 洗白时切到推理模式
                        dirty_samples = batch_x_sliced[dirty_mask]

                        purified_samples = purifier.purify(dirty_samples, noise_ratio=0.3)

                        batch_x_target[dirty_mask] = purified_samples

                        pure_weights[dirty_mask] = 0.5
                        purifier.train()


                # --- 2. 样本级重构损失 [B] ---
                # raw_recon_loss = criterion_none(outputs, batch_x_sliced)  # [B, L, C]
                raw_recon_loss = criterion_none(outputs, batch_x_target)  # [B, L, C]
                sample_recon_loss = raw_recon_loss.mean(dim=[1, 2])  # [B]

                # --- 3. 样本级记忆库熵损失 [B] ---
                # att_weights: [B, Total_P, N]
                entropy = -att_weights * torch.log(att_weights + 1e-12)
                sample_entropy_loss = entropy.mean(dim=[1, 2])  # [B]

                # --- 4. 融合与加权过滤脏数据 ---
                # 最终 Loss = 样本加权(重构 + 熵) + 净化器自身的训练Loss
                main_loss = torch.mean(pure_weights * (sample_recon_loss + 0.1 * sample_entropy_loss))
                # total_loss_per_sample = sample_recon_loss + 0.1 * sample_entropy_loss
                # loss = torch.mean(total_loss_per_sample * pure_weights)

                train_loss.append(main_loss.item())

                if (i + 1) % 100 == 0:
                    print(f"sample_recon_loss:{torch.mean(sample_recon_loss)}")
                    print(f"sample_entropy_loss:{torch.mean(sample_entropy_loss)},")
                    print(f"sample_diff_loss:{torch.mean(diff_loss)}")
                    print("\titers: {0}, epoch: {1} | main_loss: {2:.7f} | diff_loss: {3:.7f}".format(i + 1, epoch + 1, main_loss.item(), diff_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                main_loss.backward()

                model_optim.step()
                if isinstance(diff_loss, torch.Tensor) and diff_loss.requires_grad:
                    diff_loss.backward()
                    purifier_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim,None, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        # 加载测试集和训练集数据
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        # 如果为test 则从指定路径加载训练好的模型权重
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        # 初始化 attens_energy 列表用于存储能量分数
        attens_energy = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()   
        self.anomaly_criterion = nn.MSELoss(reduction='none')

        # (1) stastic on the train set 在训练集上计算重构误差作为正常样本的异常分数：
        with torch.no_grad():
            # decomp = self.model.decomposition
            # batch_x: 输入特征数据；batch_y: 对应的目标标签数据
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # 将输入数据转换为浮点型并移至指定设备
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _, _ = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                # 切片对齐计算分数,criterion计算输入与重构输出的逐元素MSE损失的均值（重构误差）作为异常分数
                # score_time = torch.mean(self.anomaly_criterion(batch_x[:, :, f_dim:], outputs[:, :, f_dim:]), dim=-1)

                # ------------------ 取误差最大的 Top-5 通道均值 ------------------
                # 这样既保留了多指标联合异常的特性，又过滤了正常通道的干扰
                mse_per_channel = self.anomaly_criterion(batch_x[:, :, f_dim:], outputs[:, :, f_dim:])  # [B, L, C]
                topk_errors, _ = torch.topk(mse_per_channel, k=5, dim=-1)  # 假设 K=5
                score_time = torch.mean(topk_errors, dim=-1)  # [B, L]
                # ==============================


                score = score_time.detach().cpu().numpy()
                attens_energy.append(score)
        # 将所有训练样本的异常分数连接成一维数组
        # 保存为训练集异常分数
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        # 在测试集上重复类似过程计算异常分数同时收集测试标签
        attens_energy = []
        test_labels = []
        test_inputs = []  # <--- 新增：用于存储原始序列

        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs, _,_,_ = self.model(batch_x)
            # criterion
            # 原时间域异常分数
            # score_time = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            f_dim = -1 if self.args.features == 'MS' else 0
            # 切片对齐
            # score_time = torch.mean(self.anomaly_criterion(batch_x[:, :, f_dim:], outputs[:, :, f_dim:]), dim=-1)
            mse_per_channel = self.anomaly_criterion(batch_x[:, :, f_dim:], outputs[:, :, f_dim:])  # [B, L, C]
            topk_errors, _ = torch.topk(mse_per_channel, k=5, dim=-1)  # 假设 K=5
            score_time = torch.mean(topk_errors, dim=-1)  # [B, L]


            score = score_time.detach().cpu().numpy()
            attens_energy.append(score)
            # 收集测试标签
            test_labels.append(batch_y)

            # <--- 新增：收集输入数据用于画图 (取最后一个通道特征) --->
            test_inputs.append(batch_x[:, :, 0].detach().cpu().numpy())

        # 将所有测试集样本的异常分数连接成一维数组
        # 保存为测试集异常分数
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # <--- 新增：将原始数据展平，与分数和标签对齐 --->
        test_inputs = np.concatenate(test_inputs, axis=0).reshape(-1)
        # 连接训练集和测试集的异常分数
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # 计算重构误差分数中阈值为（100-anomaly_ratio）的分数，并将其作为区分正常异常的阈值
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        # 将 test_energy 中大于阈值的样本标记为1（异常），小于等于阈值的标记为0（正常）
        pred = (test_energy > threshold).astype(int)
        # 将之前收集的测试集标签列表沿着第0轴（批次维度）连接成一个一维数组
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        # 将 test_labels 转换为 NumPy 数组格式（尽管它很可能已经是数组了）
        test_labels = np.array(test_labels)
        # 将测试标签转换为整数类型，作为真实标签（ground truth）
        # 这些标签指示了每个测试样本是否真的是异常点
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment 调用 adjustment 函数对检测结果进行点调整（通常用于处理连续异常点）
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        # evaluation_result2 = get_metrics(pred, gt)
        # print(f"第二次评估{evaluation_result2}")
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
