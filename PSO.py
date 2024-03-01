import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
import torch
from python_engine_demo import Engine, initial_parameter
from torch_fit import Sample_engine_data, NeuralNetwork, configure
from torch_fit import configure as torch_fit_configure
import time
import os
import tqdm
from plot_pic import plot_xy_pic
from multiprocessing import Pool
import multiprocessing
import copy
import onnxruntime as ort
import argparse
import onnx
import pandas as pd
from ONNXCombine import merge_onnx_models
from generate_data import check_current_state

mpl.rcParams['font.sans-serif'] = ['SimHei']

def compoment_Model(initial_parameter_obj, index, pred_Y, output_max, output_min):
    engine = Engine(normal_engine=False)
    # print(initial_parameter_obj.Set_Rs, initial_parameter_obj.ADD_M_fuel, initial_parameter_obj.A8, initial_parameter_obj.A9)
    ATR_State_obj = engine.reset(initial_parameter_obj) # 输入输出为实际值
    temp = np.array([ATR_State_obj.contents.Rs, ATR_State_obj.contents.Pi_t, ATR_State_obj.contents.combustor_temperature, \
        ATR_State_obj.contents.SM, ATR_State_obj.contents.Nozzle_F_ideal, ATR_State_obj.contents.Nozzle_Isp_ideal, 1 if check_current_state(engine,ATR_State_obj) else 0])
    pred_Y[index] = (temp-output_min)/(output_max-output_min)
    # print(pred_Y[index])
    # pred_Y[index] = [output[index][0], output[index][1], output[index][2], output[index][3], output[index][4], output[index][5]]


class PSO:
    def __init__(self, iter_time, size, repeat=0, test_index=0, test_data_obj=None, NN_model=False, max_F_model=True, args=None, onnx_model_path=None):
        # 最大化适应度的值
        self.args = args
        self.repeat = repeat # 第几次重复实验
        
        self.test_index = test_index # 选取excel中的第几个工况点开展测试
        self.engine = Engine(no_load_dll=False)
        if test_data_obj is None:
            self.test_data_obj = Sample_engine_data(input_index=self.args.torch_fit_input_index, output_index=self.args.torch_fit_output_index, load_npy_data=False, load_xlsx_data=True, args=self.args, train=False)
        else:
            self.test_data_obj = test_data_obj
        self.x_data, self.y_data, _ = self.test_data_obj.get_one_item(index=self.test_index)
        # print(self.x_data, self.y_data)

        self.dimension = 4  # 变量个数
        self.iter_time = iter_time  # 迭代次数
        self.size = size  # 粒子数量
        self.NN_model = NN_model # 采用神经网络计算true， 采用部件级模型计算false
        self.max_F_model = max_F_model # 最大推力模式true，最大比冲模式false

        if self.NN_model:
            self.x_low = [self.engine.min_GG_M_fuel, self.engine.min_ADD_M_fuel, self.engine.min_A8, self.engine.min_A9]
            self.x_high = [self.engine.max_GG_M_fuel, self.engine.max_ADD_M_fuel, self.engine.max_A8, self.engine.max_A9]
            self.onnx_session = ort.InferenceSession(onnx_model_path)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        else:
            self.x_low = [self.engine.min_Rs, self.engine.min_ADD_M_fuel, self.engine.min_A8, self.engine.min_A9]
            self.x_low = [0.6, self.engine.min_ADD_M_fuel, self.engine.min_A8, self.engine.min_A9]
            self.x_high = [self.engine.max_Rs, self.engine.max_ADD_M_fuel, self.engine.max_A8, self.engine.max_A9]

        self.save_path = r"test_result/PSO"
        self.str_flag = "{}_index({})_time({})_size({})_repeat({})_{}".format("maxF" if max_F_model else "maxIsp", self.test_index, self.iter_time, self.size, self.repeat, "NN" if self.NN_model else "CM")
        os.makedirs(self.save_path, exist_ok=True)
        
        # print(self.size, self.dimension)
        self.v_low = [-abs(x-y)*0.01 for x,y in zip(self.x_low, self.x_high)] # 速度变化的最小值是该变量范围的10%(负数)
        self.v_high = [abs(x-y)*0.01 for x,y in zip(self.x_low, self.x_high)] # 速度变化的最大值是该变量范围的10%(正数)
        self.v = np.random.uniform(low=self.v_low, high=self.v_high, size=(self.size, self.dimension))  # 所有粒子的速度,随机初始化
        self.x = np.random.uniform(low=self.x_low, high=self.x_high, size=(self.size, self.dimension)) # 所有粒子的位置,随机初始化
        self.output = np.zeros((self.size, len(self.test_data_obj.choice_index_y))) # 所有粒子处发动机的状态
        self.fitness_values = np.zeros(self.size) # 所有粒子的适应度值

        self.x_per_best = np.zeros((self.size, self.dimension))  # 每个粒子的最优位置
        self.output_per_best = np.zeros((self.size, len(self.test_data_obj.choice_index_y))) # 每个粒子最佳位置处发动机的状态
        self.fitness_values_per_best = np.zeros(self.size) # 每个粒子最佳适应度值

        self.x_global_best = np.zeros(self.dimension)  # 全局最优粒子的位置
        self.fitness_values_global_best = -float("inf") # 全局的最佳适应度值，初始时，适应度值为负无穷
        self.output_global_best = np.zeros(len(self.test_data_obj.choice_index_y)) # 全局最优处发动机的状态参数

        self.x_per_history = []  # 每个粒子位置的历史值
        self.output_per_history = [] # 每个粒子发动机状态参数的历史值
        self.fitness_values_per_history = [] # 每个粒子适应度值的历史值
        self.x_per_best_history = []  # 每个粒子最优位置的历史值
        self.output_per_best_history = [] # 每个粒子最优位置处发动机状态参数的历史值
        self.fitness_values_per_best_history = [] # 每个粒子最佳适应度值的历史值
        self.x_global_best_history = []  # 全局最优粒子位置的历史值
        self.output_global_best_history = [] # 全局最优处发动机状态参数的历史值
        self.fitness_values_global_best_history = [] # 全局最优粒子最佳适应度值的历史值

        self.H = self.x_data[0]   # 寻优高度
        self.Ma = self.x_data[1]  # 寻优马赫数
        self.F_ideal_limit = self.y_data[4]  # 最优比冲模式，推力目标值
        self.combustor_temperature_limit = 2300 # 加力燃烧室出口温度限制值
        self.SM_limit = 10 # 喘振裕度限制值
        self.Rs_limit = 1.0 # 转速限制值

        # 发动机模型初始参数
        self.initial_parameter_obj = initial_parameter()
        self.initial_parameter_obj.Set_altitude = self.H
        self.initial_parameter_obj.Set_Ma = self.Ma
        self.initial_parameter_obj.Set_Rs = 1.0
        self.initial_parameter_obj.ADD_M_fuel = 0.1
        self.initial_parameter_obj.A8 = 0.15
        self.initial_parameter_obj.A9 = 0.22
        self.initial_parameter_obj.Pressure_GG_Total = 1696000
        self.initial_parameter_obj.betac = 50
        self.initial_parameter_obj.betat = 50
        
        # 限制参数归一化
        self.combustor_temperature_limit_0_1 = (self.combustor_temperature_limit-self.engine.min_combustor_temperature)/(self.engine.max_combustor_temperature-self.engine.min_combustor_temperature)
        self.SM_limit_0_1 = (self.SM_limit-self.engine.min_SM)/(self.engine.max_SM-self.engine.min_SM)
        self.Rs_limit_0_1 = (self.Rs_limit-self.engine.min_Rs)/(self.engine.max_Rs-self.engine.min_Rs)
        self.F_ideal_limit_0_1 = (self.F_ideal_limit-self.engine.min_F_ideal)/(self.engine.max_F_ideal-self.engine.min_F_ideal)

        # 初始化第0代初始全局最优解
        self.fitness(self.x)
        for i in range(self.size):
            self.x_per_best[i] = self.x[i]  # 第i个例子的初始位置
            self.output_per_best[i] = self.output[i]  # 第i个例子的初始发动机状态
            self.fitness_values_per_best[i] = self.fitness_values[i]# 第i个例子的初始适应度值

            # 记录当前所有例子中最佳粒子的位置和适应度值
            if self.fitness_values[i] > self.fitness_values_global_best:
                self.x_global_best = self.x_per_best[i]
                self.output_global_best = self.output[i]
                self.fitness_values_global_best = self.fitness_values[i]
        
        self.update_history_log()

    def fitness(self, x_array):
        # 计算粒子的适应度值，一次计算所有粒子，x_array为未归一化数值
        assert len(x_array)==self.size, "维度不一致"

        if self.NN_model: # 神经网络一次计算所有粒子
            # X = np.random.uniform(0, 1, size=[self.size, 6])
            # x_data = [6.2328, 0.4320, 0.1242, 0.6259]
            # X = np.repeat([x_data], self.size, axis=0)
            X = np.insert(x_array, 0, self.H, axis=1)
            X = np.insert(X, 1, self.Ma, axis=1)
            X = np.insert(X, 6, 1.0, axis=1)
            X = np.insert(X, 7, 1.0, axis=1)
            X = np.insert(X, 8, 1.0, axis=1)
            X = np.insert(X, 9, 1.0, axis=1)
            X = np.insert(X, 10, 1.0, axis=1)
            # print(X.shape, X[0])
            # time.sleep(1000)
            X = (X-self.test_data_obj.input_min)/(self.test_data_obj.input_max-self.test_data_obj.input_min)
            pred_Y = self.onnx_session.run(None, {self.onnx_input_name: X.reshape(X.shape[0],-1).astype(np.float32) })[0]
            # print(X.shape, X[0], pred_Y[0])
            # time.sleep(1000)
            # pred_Y = self.model(torch.from_numpy(X).float().to(self.device)).detach().numpy() # 输入输出为归一化的值
        else:
            # pred_Y = np.zeros((self.size, len(self.args.torch_fit_output_index)))
            pred_Y = multiprocessing.Manager().list()
            for _ in range(self.size):
                pred_Y.append([0 for _ in self.args.torch_fit_output_index])
            p = Pool(min(os.cpu_count(), 6))
            for index in range(self.size): # 部件级模型一次计算一个粒子
                self.initial_parameter_obj.Set_Rs = x_array[index][0]
                self.initial_parameter_obj.ADD_M_fuel = x_array[index][1]
                self.initial_parameter_obj.A8 = x_array[index][2]
                self.initial_parameter_obj.A9 = x_array[index][3]
                p.apply_async(compoment_Model, (copy.deepcopy(self.initial_parameter_obj), index, pred_Y, self.test_data_obj.output_max, self.test_data_obj.output_min))
            p.close()
            p.join()

        Rs_penalty = -abs(np.clip(np.array(pred_Y)[:,0]-self.Rs_limit_0_1, 0, float("inf"))) # 超转，大于0惩罚
        combustor_temperature_penalty = -abs(np.clip(np.array(pred_Y)[:,2]-self.combustor_temperature_limit_0_1, 0, float("inf"))) # 超温，大于0惩罚
        SM_penalty = -abs(np.clip(np.array(pred_Y)[:,3]-self.SM_limit_0_1, -float("inf"), 0)) # 喘振，小于0惩罚
        F_ideal_penalty = -abs(np.array(pred_Y)[:,4]-self.F_ideal_limit_0_1) # 推力不等于目标值的惩罚
        Model_Error_penalty = np.array([0 if x[6]>0.5 else -1 for x in pred_Y]) # 推力不等于目标值的惩罚
        if self.max_F_model:
            self.fitness_values = np.array(pred_Y)[:,4] + 10*Rs_penalty + 10*combustor_temperature_penalty + 10*SM_penalty + 1000*Model_Error_penalty
        else:
            self.fitness_values = np.array(pred_Y)[:,5] + 10*F_ideal_penalty + 10*Rs_penalty + 10*combustor_temperature_penalty + 10*SM_penalty + 1000*Model_Error_penalty

        # self.fitness_values = np.array(pred_Y)[:,4] + np.array(pred_Y)[:,5] + 10*F_ideal_penalty + 10*Rs_penalty + 10*combustor_temperature_penalty + 10*SM_penalty + 1000*Model_Error_penalty

        # print(Rs_penalty[:1])
        # print(combustor_temperature_penalty[:1])
        # print(SM_penalty[:1])
        # print(F_ideal_penalty[:1])
        # print(x_array)
        # print(np.array(pred_Y)[:,4])

        self.output = copy.deepcopy(np.array(pred_Y)*(self.test_data_obj.output_max-self.test_data_obj.output_min)+self.test_data_obj.output_min)

        #  # 这是测试样例
        # for i in range(self.size):
        #     x1 = x_array[i, 0] # 第一个维度的值
        #     x2 = x_array[i, 1] # 第二个维度的值
        #     x3 = x_array[i, 2] # 第三个维度的值
        #     x4 = x_array[i, 3] # 第四个维度的值
        #     y = (x1-1)**2 + (x2+1)**2 + (x3+1)**2 + (x4+1)**2
        #     self.fitness_values[i] = -y

    def update(self, update_time):
        c1 = 2.0 # 学习因子，自身的最优值
        c2 = 2.0 # 全局的最优值
        w = 0.8 - (update_time/self.iter_time)*0.8  # 自身权重因子，惯性系数

        self.v = w*self.v + c1*random.uniform(0,1)*(self.x_per_best-self.x) + c2*random.uniform(0,1)*(self.x_global_best-self.x) # 核心公式
        self.v = np.clip(self.v, self.v_low, self.v_high)
        self.x = self.x + self.v
        self.x = np.clip(self.x, self.x_low, self.x_high)
        self.fitness(self.x)

        for i in range(self.size):
            # 更新单个例子的最优位置
            if self.fitness_values[i] > self.fitness_values_per_best[i]:
                self.x_per_best[i] = self.x[i]
                self.output_per_best[i] = self.output[i]
                self.fitness_values_per_best[i] = self.fitness_values[i]

            # 更新全局最优位置和全局最优适应度值
            if self.fitness_values[i] > self.fitness_values_global_best:
                self.x_global_best = self.x[i]
                self.output_global_best = self.output[i]
                self.fitness_values_global_best = self.fitness_values[i]
                # print(self.x_global_best, self.output_global_best)
        
        self.update_history_log()

    
    def update_history_log(self):
        # 记录学习过程历史值
        self.x_per_history.append(self.x)
        self.output_per_history.append(self.output)
        self.fitness_values_per_history.append(self.fitness_values)
        self.x_per_best_history.append(self.x_per_best)
        self.output_per_best_history.append(self.output_per_best)
        self.fitness_values_per_best_history.append(self.fitness_values_per_best)
        self.x_global_best_history.append(self.x_global_best)
        self.output_global_best_history.append(self.output_global_best)
        self.fitness_values_global_best_history.append(self.fitness_values_global_best)

    def pso(self):
        start_time = time.perf_counter()
        for update_time in tqdm.tqdm(range(self.iter_time)):
            self.update(update_time)
            # print('x_position -> {}'.format(self.x_global_best))
            # print('Value  -> {}'.format(self.fitness_values_global_best))
            # print(i)
        stop_time = time.perf_counter()

        if self.NN_model and True: # 若为神经网络优化，则优化结束后采用部件级模型，根据最优参数计算一次
            self.initial_parameter_obj.Set_Rs = self.output_global_best[0]
            self.initial_parameter_obj.ADD_M_fuel = self.x_global_best[1]
            self.initial_parameter_obj.A8 = self.x_global_best[2]
            self.initial_parameter_obj.A9 = self.x_global_best[3]
            engine = Engine(normal_engine=False)
            ATR_State_obj = engine.reset(self.initial_parameter_obj) # 输入输出为实际值
            # print(self.initial_parameter_obj.Set_Rs, self.initial_parameter_obj.ADD_M_fuel, self.initial_parameter_obj.A8, self.initial_parameter_obj.A9)
            self.CM_NN_recalc = np.array([ATR_State_obj.contents.Rs, ATR_State_obj.contents.Pi_t, ATR_State_obj.contents.combustor_temperature, \
                ATR_State_obj.contents.SM, ATR_State_obj.contents.Nozzle_F_ideal, ATR_State_obj.contents.Nozzle_Isp_ideal, 1 if check_current_state(engine, ATR_State_obj) else 0]) # 采用部件级模型重新计算的发动机状态
            # self.output_global_best = temp
            # print(ATR_State_obj.contents.GG_M_fuel)
            if not check_current_state(engine, ATR_State_obj, try_time=50):
                print("Check unpassed! \n")
        else:
            self.CM_NN_recalc = self.output_global_best
        
        CM_NN_error = (self.CM_NN_recalc - self.output_global_best)/self.CM_NN_recalc*100 # 神经网络与部件级模型的误差
        return_info = [self.test_index, self.H, self.Ma, \
            self.x_data[2], self.x_data[3], self.x_data[4], self.x_data[5], \
            self.x_global_best[0], self.x_global_best[1], self.x_global_best[2], self.x_global_best[3], \
            self.CM_NN_recalc[0], self.CM_NN_recalc[1], self.CM_NN_recalc[2], self.CM_NN_recalc[3], self.CM_NN_recalc[4], self.CM_NN_recalc[5], \
            self.output_global_best[0], self.output_global_best[1], self.output_global_best[2], self.output_global_best[3], self.output_global_best[4], self.output_global_best[5], \
            self.y_data[0], self.y_data[1], self.y_data[2], self.y_data[3], self.y_data[4], self.y_data[5],\
            CM_NN_error[0], CM_NN_error[1], CM_NN_error[2], CM_NN_error[3], CM_NN_error[4], CM_NN_error[5]]

        print("CM发动机输入值    --> H: {:.4f} - Ma: {:.4f} - {}: {:.4f} - ADD_fuel: {:.4f} - A8: {:.4f} - A9: {:.4f}".format(self.H, self.Ma, "M_fuel" if self.NN_model else "Rs", self.x_global_best[0], self.x_global_best[1], self.x_global_best[2], self.x_global_best[3]))
        print("随机发动机输入值  --> H: {:.4f} - Ma: {:.4f} - {}: {:.4f} - ADD_fuel: {:.4f} - A8: {:.4f} - A9: {:.4f}".format(self.H, self.Ma, "M_fuel" if self.NN_model else "Rs", self.x_data[2], self.x_data[3], self.x_data[4], self.x_data[5]))
        print("CM发动机状态      --> Rs: {:.4f} - Pi_t: {:.4f} - Combustor_T: {:.4f} - SM: {:.4f} - F_ideal: {:.4f} - Isp_ideal: {:.4f}".format(self.CM_NN_recalc[0], self.CM_NN_recalc[1], self.CM_NN_recalc[2], self.CM_NN_recalc[3], self.CM_NN_recalc[4], self.CM_NN_recalc[5]))
        print("NN发动机状态      --> Rs: {:.4f} - Pi_t: {:.4f} - Combustor_T: {:.4f} - SM: {:.4f} - F_ideal: {:.4f} - Isp_ideal: {:.4f}".format(self.output_global_best[0], self.output_global_best[1], self.output_global_best[2], self.output_global_best[3], self.output_global_best[4], self.output_global_best[5]))
        print("随机发动机状态    --> Rs: {:.4f} - Pi_t: {:.4f} - Combustor_T: {:.4f} - SM: {:.4f} - F_ideal: {:.4f} - Isp_ideal: {:.4f}".format(self.y_data[0], self.y_data[1], self.y_data[2], self.y_data[3], self.y_data[4], self.y_data[5]))
        print("发动机状态误差    --> Rs: {:.4f} - Pi_t: {:.4f} - Combustor_T: {:.4f} - SM: {:.4f} - F_ideal: {:.4f} - Isp_ideal: {:.4f}".format(CM_NN_error[0], CM_NN_error[1], CM_NN_error[2], CM_NN_error[3], CM_NN_error[4], CM_NN_error[5]))
        # print('Best_Fit_Value  -> {}'.format(self.fitness_values_global_best))
        print('Time -> {} s'.format(round(stop_time-start_time, 4)))
        # print((stop_time-start_time)*1)
        # print(np.array(self.x_per_history).shape)
        # print(np.array(self.output_per_history).shape)
        # print(np.array(self.fitness_values_per_history).shape)
        # print(np.array(self.x_per_best_history).shape)
        # print(np.array(self.output_per_best_history).shape)
        # print(np.array(self.fitness_values_per_best_history).shape)
        # print(np.array(self.x_global_best_history).shape)
        # print(np.array(self.output_global_best_history).shape)
        # print(np.array(self.fitness_values_global_best_history).shape)

        np.savez(r"{}/{}".format(self.save_path, self.str_flag), \
            x_per_history=np.array(self.x_per_history), \
            output_per_history=np.array(self.output_per_history), \
            fitness_values_per_history=np.array(self.fitness_values_per_history), \
            x_per_best_history=np.array(self.x_per_best_history), \
            output_per_best_history=np.array(self.output_per_best_history), \
            fitness_values_per_best_history=np.array(self.fitness_values_per_best_history), \
            x_global_best_history=np.array(self.x_global_best_history), \
            output_global_best_history=np.array(self.output_global_best_history), \
            fitness_values_global_best_history=np.array(self.fitness_values_global_best_history), \
            )
        
        per_num = 0 # 需要展示训练过程的粒子数量
        for index in range(per_num):
            self.plot_train_log(np.array(self.x_per_history)[:,self.size//per_num*index], \
                np.array(self.output_per_history)[:,self.size//per_num*index], \
                np.array(self.fitness_values_per_history)[:,self.size//per_num*index], \
                title_name="per{}".format(index))
        self.plot_train_log(np.array(self.x_global_best_history), \
            np.array(self.output_global_best_history), \
            np.array(self.fitness_values_global_best_history), \
            title_name="global")

        return return_info

    def plot_train_log(self, x_history, output_history, fitness_values_history, title_name):
        plt.figure(figsize=(12,12))
        plt.subplots_adjust(bottom=0.05, right=0.95, top=0.95, left=0.1, wspace=0.25, hspace=0.4)

        plot_xy_pic(y_data=np.array(x_history)[:,0], y_name="{}".format("M_fuel" if self.NN_model else "Rs"), subplot_index=[6,2,1])
        plot_xy_pic(y_data=np.array(x_history)[:,1], y_name="ADDF_M_fuel", subplot_index=[6,2,2])
        plot_xy_pic(y_data=np.array(x_history)[:,2], y_name="A8", subplot_index=[6,2,3])
        plot_xy_pic(y_data=np.array(x_history)[:,3], y_name="A9", subplot_index=[6,2,4])
        plot_xy_pic(y_data=np.array(output_history)[:,0], y_name="Rs", subplot_index=[6,2,5])
        plot_xy_pic(y_data=np.array(output_history)[:,1], y_name="Pi_T", subplot_index=[6,2,6])
        plot_xy_pic(y_data=np.array(output_history)[:,2], y_name="T7", subplot_index=[6,2,7])
        plot_xy_pic(y_data=np.array(output_history)[:,3], y_name="SM", subplot_index=[6,2,8])
        plot_xy_pic(y_data=np.array(output_history)[:,4], y_name="F_idle", subplot_index=[6,2,9])
        plot_xy_pic(y_data=np.array(output_history)[:,5], y_name="Isp_idle", subplot_index=[6,2,10])
        plot_xy_pic(y_data=np.array(fitness_values_history), y_name="F(y)", subplot_index=[6,2,11])

        plt.savefig(os.path.join(self.save_path, self.str_flag + "_{}.png".format(title_name)), dpi=200)
        # plt.show()
        plt.close()


def configure(random_sample=None, F_model=None):
    # 初始化参数构造器
    parser = argparse.ArgumentParser()
    # 在参数构造器中添加两个命令行参数
    parser.add_argument('--PSO_NN_model', default=False, action='store_true', help="神经网络计算")
    parser.add_argument('--PSO_max_F_mode', default=False, action='store_true', help="最大推力模式,否者最大比冲模式")
    parser.add_argument('--PSO_iter_time', type=int, default=400, help="迭代次数")
    parser.add_argument('--PSO_size', type=int, default=128, help="粒子数量")
    parser.add_argument('--PSO_repeat', type=int, default=0, help="第几次重复实验")
    parser.add_argument('--PSO_nodes', type=int, default=50, help="采用多少个节点运行")
    parser.add_argument('--PSO_node_num', type=int, default=0, help="当前计算节点编号")
    # 获取所有的命令行参数
    args, _ = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args, str_flag = torch_fit_configure(random_sample=True)
    args.torch_fit_adaboost = True # 加载excel中的所有测试点，若为False则不完全加载，加载1/5
    checkpoint_num = 299

    PSO_args = configure()
    
    if not PSO_args.PSO_NN_model: # 神经网络or部件级模型计算
        args.torch_fit_input_index = [0, 1, 2, 16, 17, 18] # 部件级模型计算有用（高度、马赫数、转速、加力流量、A8、A9）
        onnx_model_path = None
    else:
        args.torch_fit_input_index = [0, 1, 15, 16, 17, 18, 22, 23, 24, 25, 26] # 神经网络计算有用（高度、马赫数、主燃油流量、加力流量、A8、A9）
        onnx_model_path = os.path.join(r"checkpoints/fit_model/{}".format(str_flag), "merged_model_{}.onnx".format(checkpoint_num))
        print("{} ONNX Model Path-> ".format(time.strftime("%Y-%m-%d %H:%M:%S")), onnx_model_path)
    args.torch_fit_output_index = [2, 3, 10, 19, 20, 21, 47] # （转速、压比、燃烧室出口温度、喘振裕度、推力、比冲、发动机模型求解状态）
    test_data_obj = Sample_engine_data(input_index=args.torch_fit_input_index, output_index=args.torch_fit_output_index, load_npy_data=False, load_xlsx_data=True, args=args, train=False)
    print("{} Test samples num-> ".format(time.strftime("%Y-%m-%d %H:%M:%S")), test_data_obj.total_num)
    
    PSO_result = []
    total_test_index = range(0, test_data_obj.total_num, 1)
    selector_test_index = [total_test_index[x] for x in range(PSO_args.PSO_node_num, len(total_test_index), PSO_args.PSO_nodes)]
    print(selector_test_index)
    for test_index in selector_test_index:
        pso = PSO(PSO_args.PSO_iter_time, PSO_args.PSO_size, test_index=test_index, repeat=PSO_args.PSO_repeat, test_data_obj=test_data_obj, NN_model=PSO_args.PSO_NN_model, max_F_model=PSO_args.PSO_max_F_mode, args=args, onnx_model_path=onnx_model_path)
        return_info = pso.pso()
        PSO_result.append(return_info)
    
    columns = ["Test_index", "H", "Ma", \
        "original_M_fuel" if PSO_args.PSO_NN_model else "original_Rs", "original_ADD_fuel", "original_A8", "original_A9", \
        "PSO_M_fuel" if PSO_args.PSO_NN_model else "PSO_Rs", "PSO_ADD_fuel", "PSO_A8", "PSO_A9", \
        "CM_Rs", "CM_Pi_t", "CM_Combustor_T", "CM_SM", "CM_F_ideal", "CM_Isp_ideal", \
        "NN_Rs", "NN_Pi_t", "NN_Combustor_T", "NN_SM", "NN_F_ideal", "NN_Isp_ideal", \
        "original_CM_Rs", "original_Pi_t", "original_Combustor_T", "original_SM", "original_F_ideal", "original_Isp_ideal", \
        "error_CM_NN_Rs", "error_CM_NN_Pi_t", "error_CM_NN_Combustor_T", "error_CM_NN_SM", "error_CM_NN_F_ideal", "error_CM_NN_Isp_ideal"]
    df = pd.DataFrame(np.array(PSO_result), columns=columns)
    df.to_excel(r"{}/{}_time({})_size({})_nodes({})_repeat({})_{}.xlsx".format(pso.save_path, "maxF" if PSO_args.PSO_max_F_mode else "maxIsp", PSO_args.PSO_iter_time, PSO_args.PSO_size, PSO_args.PSO_node_num, PSO_args.PSO_repeat, "NN" if PSO_args.PSO_NN_model else "CM"))