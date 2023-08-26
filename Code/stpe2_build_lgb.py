#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 20:21
# @Author  : Feng Zhanpeng
# @File    : step2_build_lgb.py
# @Software: PyCharm

"""
LightGBM构建贷前申请评分卡模型代码执行顺序：
一、基于训练集样本建模
1. 加载建模过程中需要使用到的函数，因代码较多，相关函数不进行展示，具体可见本书附件，附件中对应的文件名为step1_calculate_fun.py；
2. 加载Python包；
3. 数据及文件输出路径设置，在实操过程中相关路径都要换成自己本地的路径；
4. 读入数据并进行数据预处理；
5. 将数据拆分为训练集和测试集；
6. 对训练集数据进行描述性统计分析；
7. 基于LightGBM和贝叶斯搜索算法构建最优模型；
8. 输出训练集样本的预测概率并将概率值转为模型评分；
9. 模型在训练集效果评估；

二、模型在测试集上的效果评估
10. 对测试集数据进行描述性统计分析；
11. 输出测试集样本的预测概率并将概率值转为模型评分；
12. 模型在测试集效果评估；
13. 绘制模型效果评估相关图；

三、对全量样本打分，并输出打分结果，供策略分析使用
14.加载全量需要打分的数据并进行数据预处理；
15.加载模型训练结果对应的pkl文件；
16.输出全量样本的预测概率并将概率值转为模型评分；
17.输出打分数据，供策略进一步分析使用
"""

'''
一、基于训练集样本建模
'''

# 2.加载Python包

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import *
# from step1_calculate_fun import *
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
# 需要注意的是贝叶斯优化Python包的安装语句为：pip install bayesian-optimization
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import joblib
import datetime
import os
import copy
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 3.数据及文件输出路径设置，在实操过程中相关路径都要换成自己本地的路径

# 建模数据及数据对应的数据字典存放路径
path = r'D:\\金融风控\\Python金融风控策略实践 代码和数据样例\\Chapter9\\9.2.2贷前申请评分卡模型开发\\'
# 模型运行过程中需要保存的pkl文件存储路径
path_pkl = path + 'pkl\\'
# 模型运行过程中输出的文件存储路径
path_report = path + 'report\\'

if not os.path.exists(path_pkl):
    os.makedirs(path_pkl)
if not os.path.exists(path_report):
    os.makedirs(path_report)

# 4.读入数据并进行数据预处理
# 读入数据字典
var_dict = pd.read_excel(path + "数据字典.xlsx")
var_dict.变量名 = var_dict.变量名.map(lambda x: str(x).lower().replace('\t', ''))

# 读入建模数据，并进行数据预处理
f = open(path + 'model_data.csv', encoding='utf-8')
my_data = pd.read_csv(f)

# 筛选申请时间在2020年3月到2022年1月且用信且目标字段成熟的样本进行建模
my_data = my_data[
    (my_data['apply_mth'].map(lambda x: x >= '2020-03' and x <= '2022-01')) & (my_data['if_loan_flag'] == 1) & (
                my_data['agr_mob12_dpd_30'] == 1)]

# 删除建模时不需要的字段
my_data.drop(labels=['apply_day', 'apply_week', 'apply_mth', 'product_name', 'apply_refuse_flag', 'agr_mob3_dpd_30',
                     'agr_mob6_dpd_30', 'agr_mob9_dpd_30',
                     'agr_fpd_15', 'mob3_dpd_30_act', 'mob6_dpd_30_act', 'mob9_dpd_30_act', 'fpd_15_act',
                     'if_loan_flag', 'if_loan_in_30', 'agr_mob12_dpd_30'], axis=1, inplace=True)

# 剔除灰样本
my_data = my_data[my_data['mob12_dpd_30_act'].map(lambda x: x in [0, 1])]

# 将可能出现的异常值置为缺失
for i in my_data.columns[my_data.dtypes != 'object']:
    my_data[i][my_data[i].map(lambda x: x in (-999, -9999, -9998, -999999, -99999, -998, -997, -9997))] = np.nan

for i in my_data.columns[my_data.dtypes == 'object']:
    my_data[i] = my_data[i].map(lambda x: str(x).strip())
    my_data[i][my_data[i].map(
        lambda x: x in ['-999', '-9999', '-9998', '-999999', '-99999', '-998', '-997', '-9997'])] = np.nan
    try:
        my_data[i] = my_data[i].astype('float64')
    except:
        pass

# 5.将数据拆分为训练集和测试集
trainData, testData = train_test_split(my_data, test_size=0.3, random_state=1111)

# 6.对训练集数据进行描述性统计分析
train_var_describe = describe_stat_ana(describe_data=trainData, var_dict=var_dict, sample_range='202003-202201',
                                       target='mob12_dpd_30_act', seq=1, na_threshold=0.85, vardict_varengname='变量名',
                                       vardict_varchiname='变量描述', sample_category='Train')

# 7.基于LightGBM和贝叶斯搜索算法构建最优模型
x_vars = train_var_describe['Characteristic'].tolist()
joblib.dump(x_vars, filename=path_pkl + 'x_vars.pkl')

X_train = trainData[x_vars]
y_train = trainData['mob12_dpd_30_act']
X_test = testData[x_vars]
y_test = testData['mob12_dpd_30_act']

param = {'task': 'train', 'objective': 'binary', 'boosting_type': 'rf', 'metric': {'auc'}}
param1 = copy.deepcopy(param)

# 基于贝叶斯搜索从以下参数对应的区间内寻找最优超参数
params = {
    'learning_rate': (0.005, 0.3),  # 学习速率
    'num_leaves': (4, 10),  # 叶子节点数
    'max_depth': (3, 8),  # 决策树最大深度
    'feature_fraction': (0.5, 0.8),  # 建树的特征选择比例
    'bagging_fraction': (0.5, 0.8),  # 建树的样本采样比例
    'min_gain_to_split': (0.01, 1.5),  # 决策树进行分裂的最小增益
    'lambda_l1': (0, 0.1),  # L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择，取值范围[0,1]
    'lambda_l2': (0, 0.8),  # L2正则化防止模型过拟合，一定程度上L1也可以防止过拟合
    'bagging_freq': (1, 10),  # bagging的次数，默认取值为0，取值为k表示执行k次bagging
    'min_data_in_leaf': (60, 300),  # 叶节点的最小样本数，设置的较大则树不会生长的很深，可能造成模型欠拟合
    'min_sum_hessian_in_leaf': (0.1, 1),  # 与min_data_in_leaf作用类似，可以用来处理过度拟合
    'max_bin': (80, 180)  # 特征值将被放入的最大箱数
}

# 调用贝叶斯优化函数确定最优超参数
best_params = BayesianSearch(f=lgb_evaluate, pbounds=params, init_points=30, n_iter=6)

# 确定最优模型结果对应的参数
for i, j in best_params['params'].items():
    if i in ['num_leaves', 'max_depth', 'bagging_freq', 'min_data_in_leaf', 'max_bin']:
        j = int(round(j))
    param1[i] = j

# 输出最终建模参数
print(param1)

# 基于最优参数重新建模
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
bst = lgb.train(param1, lgb_train, valid_sets=[lgb_test], num_boost_round=100) #, early_stopping_rounds=5)

# 保存建模过程
joblib.dump(bst, filename=path_pkl + 'model_result.pkl')

# 建模变量重要性评估，不建议重要性得分太高的变量入模，若变量不稳定，模型结果容易不稳定，
import_vars = pd.DataFrame({
    'Characteristic': [i.replace('_br_encoding', '') for i in bst.feature_name()],
    'Importance': bst.feature_importance(importance_type='split'), }).sort_values(by='Importance', ascending=False)

import_vars = \
pd.merge(import_vars, train_var_describe[['Characteristic', 'Description']], on='Characteristic', how='left')[
    ['Characteristic', 'Description', 'Importance']]
import_vars.head()

# 8. 输出训练集样本的预测概率并将概率值转为模型评分

# 预测训练集样本逾期概率
X_train['prob'] = bst.predict(X_train, num_iteration=bst.best_iteration)
X_train['mob12_dpd_30_act'] = y_train
X_train['pred'] = [1 if p > 0.5 else 0 for p in X_train['prob']]

# 将概率值转为评分
X_train['Odds'] = np.log(X_train['prob'] / (1 - X_train['prob']))
X_train['Score'] = 501.8622 - 28.8539 * X_train['Odds']
X_train['Score_int'] = X_train['Score'].map(lambda x: round(x))

# 将模型分限制在300-800分之间
X_train['Score_int'] = X_train['Score_int'].map(lambda x: np.clip(x, 300, 800))

# 9.模型在训练集效果评估

train_model_evaluate = model_evaluate_fun(sample='Train(202003-202201)', data=X_train, prob1='prob', pred1='pred',
                                          target='mob12_dpd_30_act')

'''
二、模型在测试集上的效果评估
'''

# 10.对测试集数据进行描述性统计分析

test_var_describe = describe_stat_ana(describe_data=testData, var_dict=var_dict, sample_range='202003-202201',
                                      target='mob12_dpd_30_act', seq=1, na_threshold=0.85, vardict_varengname='变量名',
                                      vardict_varchiname='变量描述', sample_category='Test')

# 合并训练集和测试集数据描述性统计分析结果，可基于该结果查看训练和测试样本的差异性
all_var_describe = pd.concat([train_var_describe, test_var_describe])

# 11. 输出测试集样本的预测概率并将概率值转为模型评分
X_test['prob'] = bst.predict(X_test, num_iteration=bst.best_iteration)  # 输出的是概率结果
X_test['mob12_dpd_30_act'] = testData['mob12_dpd_30_act']

X_test['pred'] = [1 if p > 0.5 else 0 for p in X_test['prob']]
X_test['Odds'] = np.log(X_test['prob'] / (1 - X_test['prob']))
X_test['Score'] = 501.8622 - 28.8539 * X_test['Odds']
X_test['Score_int'] = X_test['Score'].map(lambda x: round(x))
X_test['Score_int'] = X_test['Score_int'].map(lambda x: np.clip(x, 300, 800))

# 12.模型在测试集效果评估

test_model_evaluate = model_evaluate_fun(sample='Test(202003-202201)', data=X_test, prob1='prob', pred1='pred',
                                         target='mob12_dpd_30_act')
# 合并模型评估统计量
train_model_evaluate_trans = pd.DataFrame.from_dict(train_model_evaluate, orient='index').T
test_model_evaluate_trans = pd.DataFrame.from_dict(test_model_evaluate, orient='index').T
all_model_evaluate = pd.concat([train_model_evaluate_trans, test_model_evaluate_trans])

# 13.绘制模型效果评估相关图

# ROC曲线
plot_roc_fun(train_data=X_train, test_data=X_test, train_label='Train', test_label='Test', prob='prob',
             target='mob12_dpd_30_act', dpi=150, bg_color='white', linewidth=0.9,
             save_name=path_report + 'all_roc_curve.png')

# PR曲线
plot_pr_fun(train_data=X_train, test_data=X_test, train_label='Train', test_label='Test', prob='prob',
            target='mob12_dpd_30_act', dpi=150, bg_color='white', linewidth=0.9,
            save_name=path_report + 'all_pr_curve.png')

# KS曲线
plot_ks_combine_fun(train_data=X_train, test_data=X_test, train_label='Train Set KS Curve',
                    test_label='Test Set KS Curve', score='Score', target='mob12_dpd_30_act', dpi=150, bg_color='white',
                    bins=135, linewidth=0.9, save_name=path_report + 'all_ks_curve.png')

# 好坏样本对应的模型得分分布图
black_white_dis_combine_plot(train_data=X_train, test_data=X_test, train_label='Train Set Good and Bad distribution',
                             test_label='Test Set Good and Bad Distribution', score='Score', target='mob12_dpd_30_act',
                             dpi=120, bg_color='white', bins=160,
                             save_name=path_report + 'all_model_score_dis_curve.png')

'''
三、对全量样本打分，并输出打分结果，供策略分析使用
'''

# 14.加载全量需要打分的数据并进行数据预处理
f = open(path + 'model_data.csv', encoding='utf-8')
my_data = pd.read_csv(f)

# 将可能出现的异常值置为缺失
for i in my_data.columns[my_data.dtypes != 'object']:
    my_data[i][my_data[i].map(lambda x: x in (-999, -9999, -9998, -999999, -99999, -998, -997, -9997))] = np.nan

for i in my_data.columns[my_data.dtypes == 'object']:
    my_data[i] = my_data[i].map(lambda x: str(x).strip())
    my_data[i][my_data[i].map(
        lambda x: x in ['-999', '-9999', '-9998', '-999999', '-99999', '-998', '-997', '-9997'])] = np.nan
    try:
        my_data[i] = my_data[i].astype('float64')
    except:
        pass

# 15.加载模型训练结果对应的pkl文件

bst = joblib.load(path_pkl + 'model_result.pkl')

# 获取入模变量
var_in_model_init = bst.feature_name()
X_valid = my_data[var_in_model_init]

# 16.输出全量样本的预测概率并将概率值转为模型评分

X_valid['prob'] = bst.predict(X_valid, num_iteration=bst.best_iteration)  # 输出的是概率结果
X_valid['pred'] = [1 if p > 0.5 else 0 for p in X_valid['prob']]
X_valid['Odds'] = np.log(X_valid['prob'] / (1 - X_valid['prob']))
X_valid['Score'] = 501.8622 - 28.8539 * X_valid['Odds']
X_valid['Score_int'] = X_valid['Score'].map(lambda x: round(x))
X_valid['Score_int'] = X_valid['Score_int'].map(lambda x: np.clip(x, 300, 800))

# 17.输出打分数据，供策略进一步分析使用

output_data = pd.merge(my_data, X_valid[['prob', 'Score_int']], left_index=True, right_index=True, how='inner')
output_data.to_csv(path + 'model_score_data.csv', index=False)
