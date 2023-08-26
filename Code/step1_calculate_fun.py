#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 20:21
# @Author  : Feng Zhanpeng
# @File    : step1_calculate_fun.py
# @Software: PyCharm
import datetime
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


# 变量描述性统计分析函数
def describe_stat_ana(describe_data,var_dict,sample_range='201805-201806',target='fpd_10_act',seq=1,na_threshold=0.9,
                      vardict_varengname='变量名',vardict_varchiname='变量描述',sample_category='训练集'):
    '''
    :param describe_data:         需要分析的数据框
    :param var_dict:              数据字典
    :param sample_range:          样本区间
    :param target:                目标字段
    :param seq:                   分析变量从序号seq开始算起，如seq=1，表示分析的第一个变量序号是1，分析的第二个变量序号是2
    :param na_threshold:          缺失值占比大于na_threshold，则会在后续的建模中剔除该变量
    :param vardict_varengname:    数据字典中，变量英文名对应的列名
    :param vardict_varchiname:    数据字典中，变量中文名对应的列名
    :param sample_category:       用来区分训练 、测试 、验证集
    :return:                      变量的描述性统计分析结果
    '''
    var_detail = pd.DataFrame(columns=["Sample","Sequence", "Ana_time", "Characteristic", "Description", "变量类别",
                          "样本区间", "坏客户定义", "%Bad_Rate","总样本量", "缺失量", "缺失率", "变量取值数（包含缺失值）", "变量取值数（不含缺失值）",
                            "单一值最大占比的变量值", "单一值最大占比的样本量", "单一值最大占比", "单一值第二大占比的变量值", "单一值第二大占比的样本量",
                            "单一值第二大占比", "单一值第三大占比的变量值", "单一值第三大占比的样本量", "单一值第三大占比",
                            "单一值前二大占比的总样本量", "单一值前二大占比总和", "单一值前三大占比的总样本量","单一值前三大占比总和","异众比率",
                            "最大值","最大值数量","最大值占比","最小值","最小值数量","最小值占比","极差",
                            "平均值","下四分位数","中位数","上四分位数",  "标准差", "离散系数","偏态系数",'峰态系数',"标签1"])
    for var in describe_data.columns[:-1]:
        print('正在分析第',seq,'个变量:',var)
        sample=sample_category
        seq=seq
        ana_time=datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        var_english_name=var
        var1=var
        var_chinese_name=var_dict[vardict_varchiname][var_dict[vardict_varengname]==var1].values[0] if sum(var_dict[vardict_varengname]==var1) else None
        var_category='分类数据' if  sum(describe_data[[var]].dtypes=='object')>0 else '数值型数据'
        sample_range=sample_range
        bad_define=target
        # 计算badrate,包含和不包含缺失值
        data_nona = describe_data[[var, target]].dropna()
        data_withna = describe_data[[var, target]]
        bad_rate_withna = data_withna[target].value_counts(normalize=True)[1]
        total_cnt = len(data_withna)
        na_cnt = describe_data[var].isnull().sum()
        na_rate = na_cnt * 1.0 / total_cnt
        unique_cnt_withna = len(data_withna[var].unique())
        unique_cnt_nona = len(data_nona[var].unique())
        # 前3大占比值分析
        first_info_rate = data_withna[var].value_counts(dropna=False, normalize=True).sort_values(ascending=False)
        first_info_cnt = data_withna[var].value_counts(dropna=False).sort_values(ascending=False)
        max_cnt_value = first_info_rate.index[0]
        max_cnt_value_num = first_info_cnt.tolist()[0]
        max_cnt_value_rate = first_info_rate.max()
        second_cnt_value = first_info_rate.index[1] if len(first_info_rate) > 1 else np.nan
        second_cnt_value_num = first_info_cnt.tolist()[1] if len(first_info_cnt) > 1 else np.nan
        second_cnt_value_rate = first_info_rate.tolist()[1] if len(first_info_rate) > 1 else np.nan
        second_cnt_value_rate_01 = first_info_rate.tolist()[1] if len(first_info_rate) > 1 else np.nan
        third_cnt_value = first_info_rate.index[2] if len(first_info_rate) > 2 else np.nan
        third_cnt_value_num = first_info_cnt.tolist()[2] if len(first_info_cnt) > 2 else np.nan
        third_cnt_value_rate = first_info_rate.tolist()[2] if len(first_info_rate) > 2 else np.nan
        third_cnt_value_rate_01 = first_info_rate.tolist()[2] if len(first_info_rate) > 2 else np.nan
        var_max_cnt_value_12num = np.nansum([max_cnt_value_num, second_cnt_value_num])
        var_max_cnt_value_12rate = np.nansum([first_info_rate.tolist()[0], second_cnt_value_rate_01])
        var_max_cnt_value_123num = np.nansum([max_cnt_value_num, second_cnt_value_num, third_cnt_value_num])
        var_max_cnt_value_123rate = np.nansum([first_info_rate.tolist()[0], second_cnt_value_rate_01, third_cnt_value_rate_01])
        var_not_mode_value_rate = 1- max_cnt_value_rate
        if var_category == '数值型数据':
            max_value = data_nona[var].max()
            max_value_num = sum(data_nona[var] == max_value)
            max_value_rate = sum(data_nona[var] == max_value) / total_cnt if total_cnt>0 else np.nan
            min_value = data_nona[var].min()
            min_value_num = sum(data_nona[var] == min_value)
            min_value_rate = sum(data_nona[var] == min_value) / total_cnt if total_cnt>0 else np.nan
            range_value = max_value - min_value
            mean_value = data_nona[var].mean()
            q1_value = np.percentile(data_nona[var], 25) if len(data_nona[var])>0 else np.nan
            median_value = data_nona[var].median()
            q3_value = np.percentile(data_nona[var], 75) if len(data_nona[var])>0 else np.nan
            std_value = np.std(data_nona[var])
            cv = std_value / mean_value if mean_value != 0 else 0
            skew = stats.skew(data_nona[var])  # 去除缺失值后计算偏度和峰度
            kurtosis = stats.kurtosis(data_nona[var])
        if var_category == '分类数据':
            data_ls = data_nona[var].value_counts().reset_index()
            max_value = data_ls['index'][data_ls[var] == data_ls[var].max()].values[0] if len(data_ls) > 0 else np.nan
            max_value_num = sum(data_nona[var] == max_value)
            max_value_rate = sum(data_nona[var] == max_value) / total_cnt if total_cnt>0 else np.nan
            min_value = data_ls['index'][data_ls[var] == data_ls[var].min()].values[0] if len(data_ls) > 0 else np.nan
            min_value_num = sum(data_nona[var] == min_value)
            min_value_rate = sum(data_nona[var] == min_value) / total_cnt if total_cnt>0 else np.nan
            range_value = np.nan
            mean_value = np.nan
            q1_value = np.nan
            median_value = np.nan
            q3_value = np.nan
            std_value = np.nan
            cv = np.nan
            skew = np.nan
            kurtosis = np.nan
        na_rate1 = na_cnt / total_cnt
        if (na_rate1 <= na_threshold) & (unique_cnt_nona >= 2):
            if_model_choose = 'Y'
        else:
            if_model_choose = 'N'
        sum_info = [sample, seq, ana_time, var_english_name, var_chinese_name, var_category,
                    sample_range, bad_define, bad_rate_withna, total_cnt, na_cnt, na_rate, unique_cnt_withna,
                    unique_cnt_nona,max_cnt_value, max_cnt_value_num, max_cnt_value_rate, second_cnt_value, second_cnt_value_num,
                    second_cnt_value_rate,third_cnt_value, third_cnt_value_num, third_cnt_value_rate, var_max_cnt_value_12num,
                    var_max_cnt_value_12rate, var_max_cnt_value_123num, var_max_cnt_value_123rate,var_not_mode_value_rate,
                    max_value, max_value_num, max_value_rate,min_value, min_value_num, min_value_rate,range_value,
                    mean_value, q1_value, median_value, q3_value, std_value, cv,skew,kurtosis, if_model_choose]
        sum_info01 = pd.DataFrame(sum_info).T
        sum_info01.columns = var_detail.columns.tolist()
        var_detail = var_detail._append(sum_info01, ignore_index=True)
        # var_detail = pd.concat([var_detail, pd.DataFrame([sum_info01])], ignore_index=True)
        seq += 1
    return (var_detail)


# 贝叶斯优化函数

def BayesianSearch(f, pbounds,init_points,n_iter):
    '''贝叶斯优化器
    :param f:  目标函数
    :param pbounds: 搜索区间  pbounds = {'x': (2, 4), 'y': (-3, 3)}
    :param init_points:  您想要执行多少个随机探索步骤。 随机探索可以通过多样化勘探空间来提供帮助。
    :param n_iter: 要执行的贝叶斯优化步骤数。 越多的步骤越有可能找到一个好的最大值。
    :return:
    '''
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(f, pbounds)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=n_iter)
    # 输出结果
    print(bayes)
    best_params = bayes.max
    print(best_params)
    return best_params

'''
Lightgbm参数说明
避免过拟合的参数
使用较小的 max_bin
使用较小的 num_leaves
使用较大的 min_data_in_leaf 和 min_sum_hessian_in_leaf(默认的参数是1e-3，这个参数越大，则泛化能力越好；越小则叶子结点越纯粹，越容易过拟合。)
通过设置 bagging_fraction 和 bagging_freq 来使用 bagging
通过设置feature_fraction 来使用特征子抽样
使用更大的训练数据
设置正则化参数,lambda_l1,lambda_l2,增大min_gain_to_split，可防止微小的增益分裂
尝试 max_depth 来避免生成过深的树
'''

def lgb_evaluate(learning_rate=0.1, num_leaves=31, max_depth=-1, feature_fraction=1, bagging_fraction=1,min_gain_to_split=0,\
                 lambda_l1=0,lambda_l2=0,bagging_freq=3,min_data_in_leaf=50,min_sum_hessian_in_leaf=0,max_bin=200):
    ''' 自定义的模型评估函数    boosting_type='gbdt'# 提升树的类型 gbdt,dart,goss,rf ,默认为gbdt；  boosting_type='rf'时，需要设置subsample_freq参数
    :param learning_rate:  学习率
    :param num_leaves:树的最大叶子数，对比xgboost一般为2^(max_depth)
    :param max_depth:
    :param feature_fraction:
    :param bagging_fraction:
    :param lambda_l1:
    :param lambda_l2:# 越小l2正则程度越高
    :param objective:# 'binary', # 目标函数
    :param metric:#  {'auc'} binary_logloss
    :param boosting_type:  # 设置提升类型
    :return:
    '''
    # 贝叶斯优化器生成的超参数
    param = {}
    param['learning_rate'] = learning_rate
    param['num_leaves'] = int(num_leaves)
    param['max_depth'] = int(max_depth)
    param['feature_fraction'] = feature_fraction
    param['bagging_fraction'] = bagging_fraction
    param['min_gain_to_split'] = min_gain_to_split
    param['lambda_l1'] = lambda_l1
    param['lambda_l2'] = lambda_l2
    param['bagging_freq'] = int(bagging_freq)
    param['min_data_in_leaf'] = int(min_data_in_leaf)
    param['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
    param['max_bin'] = int(max_bin)
    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # 注意BayesianOptimization会向最大评估值的方向优化; 若目标函数为损失函数 ，则需要取负值
    gbm = lgb.train(param, lgb_train, valid_sets=[lgb_test], num_boost_round=150) #, early_stopping_rounds=10)
    predictions = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    score = roc_auc_score(y_test, predictions)
    return score

# 计算模型KS值
def KS_calculate(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    if np.array(df[score])[0]<1.1 :
        total = df.groupby([score])[target].count()
        bad = df.groupby([score])[target].sum()
        all = pd.DataFrame({'total':total, 'bad':bad})
        all['good'] = all['total'] - all['bad']
        all[score] = all.index
        all.index = range(len(all))
        all = all.sort_values(by=score,ascending=False)
        all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
        all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
        KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
        return max(KS)
    if np.array(df[score])[0]>1 :
        total = df.groupby([score])[target].count()
        bad = df.groupby([score])[target].sum()
        all = pd.DataFrame({'total':total, 'bad':bad})
        all['good'] = all['total'] - all['bad']
        all[score] = all.index
        all.index = range(len(all))
        all = all.sort_values(by=score,ascending=True)
        all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
        all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
        KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
        return max(KS)

# 计算模型的Auc、Ks等指标
def model_evaluate_fun(sample,data,prob1,pred1,target):
    '''
    :param sample: 模型样本类型   Train  test   valid
    :param data:  数据
    :param prob:  预测的概率值
    :param pred:  预测的类别值
    :param target: 目标字段
    :return:
    '''
    Obs=len(data)
    Bad_Rate=data[target].mean()
    Min_Prob = data[prob1].min()
    Max_Prob = data[prob1].max()
    Ks_value = KS_calculate(data, prob1,target)
    Auc_value = roc_auc_score(data[target], data[prob1])
    print('Sample:' + str(sample) + '\n' + 'Obs=%s' % str(Obs) + '\n' + 'Bad_Rate=%s' % '{:.2f}%'.format(Bad_Rate) + '\n' +
          'Min_Prob=%s' % str(round(Min_Prob, 2)) + '\n'  + 'Max_Prob=%s' % str(round(Max_Prob, 2)) + '\n' 
          'KS=%s' % str(round(Ks_value, 2)) + '\n' + 'Auc=%s' % str(round(Auc_value, 2)))
    return ({'Sample': sample, '#Obs': Obs, '%Bad_rate': '{:.2f}%'.format(Bad_Rate),'Min_Prob':str(round(Min_Prob, 2)),
             'Max_Prob':str(round(Max_Prob, 2)),'KS': str(round(Ks_value, 2)), 'AUC': str(round(Auc_value, 2))})

# ROC曲线
def plot_roc_fun(train_data,test_data,train_label,test_label,prob,target,dpi,bg_color,linewidth,save_name):
    '''
    :param data: df
    :param prob: 预测的概率值
    :param target: 目标字段
    :param figsize:  图片大小
    :param dpi:  图片像素
    :param bg_color:  背景色
    :param mian:  图片标题
    :param save_name:  图片存储路径和名称
    :return:
    '''
    plt.grid(axis="y", alpha=0.6)  # 画y轴网格线
    plt.figure(1, dpi=dpi)
    # 设置figure窗体在颜色
    plt.rcParams['figure.facecolor'] = bg_color
    # 设置axes绘图区的颜色
    plt.rcParams['axes.facecolor'] = bg_color
    if len(test_data) > 10:
        fpr, tpr, thresholds = roc_curve(train_data[target], train_data['prob'])
        fpr1, tpr1, thresholds1 = roc_curve(test_data[target], test_data['prob'])
        Auc_train = roc_auc_score(train_data[target], train_data[prob])
        Auc_test = roc_auc_score(test_data[target], test_data[prob])
        plt.plot(fpr, tpr, label=train_label,linewidth=linewidth,color='#4F81BD')
        plt.plot(fpr1, tpr1, label=test_label,linewidth=linewidth,color='#8064A2')
        plt.title('ROC Curve' + '\n' + 'Train_AUC=' + str(round(Auc_train, 2))+',Test_AUC='+ str(round(Auc_test, 2)))
    else:
        fpr, tpr, thresholds = roc_curve(train_data[target], train_data['prob'])
        Auc_train = roc_auc_score(train_data[target], train_data[prob])
        plt.plot(fpr, tpr, label=train_label,linewidth=linewidth,color='#4F81BD')
        plt.title('ROC Curve' + '\n' + 'Train_AUC=' + str(round(Auc_train, 2)))
    plt.plot([0, 1], [0, 1],label='Random',linewidth=linewidth,linestyle='--',color='#C0504D')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='upper left')
    plt.savefig(save_name)
    plt.close()
    #plt.clf()

## KS曲线
def plot_ks_combine_fun(train_data,test_data,train_label,test_label,score,target,dpi,bg_color,bins,linewidth,save_name):
    '''
    :param train_data:
    :param test_data:
    :param train_label:  训练集图片标题
    :param test_label:
    :param score:  模型分
    :param target:  目标字段
    :param dpi: 图片像素
    :param bg_color:  背景色
    :param bins:  直方图分箱数
    :param linewidth:  线条宽度
    :param save_name:  保存图片名称
    :return:
    '''
    if len(test_data) > 10:
        plt.figure(1,figsize=(13, 5))
        plt.subplot(121)
        plot_ks_fun(data=train_data, score=score, target=target,dpi=dpi, bg_color=bg_color, bins=bins,main=train_label, linewidth=linewidth)
        plt.subplot(122)
        plot_ks_fun(data=test_data, score=score, target=target, dpi=dpi, bg_color=bg_color, bins=bins, main=test_label,linewidth=linewidth)
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()
        #plt.clf()
    else:
        plt.figure(1,figsize=(7, 5))
        plot_ks_fun(data=train_data, score=score, target=target,dpi=dpi,  bg_color=bg_color, bins=bins,main=train_label, linewidth=linewidth)
        plt.savefig(save_name)
        plt.close()
        #plt.clf()

def plot_ks_fun(data,score,target,dpi,bg_color,bins,main,linewidth):
    '''
    :param data:
    :param score:  转换刻度后的模型评分
    :param target:  目标字段
    :param figsize: 图片大小
    :param dpi:  像素
    :param bg_color:  背景
    :param bins:  直方图分bin数
    :param mian: 标题
    :param save_name: 保存名称
    :return:
    '''
    plt.grid(axis="y",alpha=0.6)  # 画y轴网格线
    total = data.groupby(score)[target].count()
    bad = data.groupby(score)[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all['train_score'] = all.index
    all = all.sort_values(by='train_score',ascending=True)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['KS'] = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    KS_value = max(all['KS'])
    pop=all[all.index<= all[all['KS'] == KS_value].index[0]].total.sum()/len(data)
    KS_cutoff=all.train_score[all[all['KS'] == KS_value].index[0]]
    plt.figure(1,dpi=dpi)
    # 设置figure窗体在颜色
    plt.rcParams['figure.facecolor'] = bg_color
    # 设置axes绘图区的颜色
    plt.rcParams['axes.facecolor'] = bg_color
    plt.hist(all.train_score, bins=bins, density=True, edgecolor='#696969', facecolor='#1D364B', alpha=0.6)
    plt.xlabel('Score')
    plt.ylabel('%Obs')
    plt.twinx()
    plt.plot(all.train_score, all['badCumRate'], label='%Cummulative_Bad', linewidth=linewidth, color='#4F81BD')
    plt.plot(all.train_score, all['goodCumRate'], label='%Cummulative_Good', linewidth=linewidth, color='#9BBB59')
    plt.plot(all.train_score, all['KS'], label='KS_Value', linewidth=linewidth, color='#C0504D')
    plt.ylabel('%Cummulative Obs')
    plt.title(main + '\n' +'KS=%s' % round(KS_value, 2) + ' at %Cum_Obs=' + str(
        round(pop * 100, 2)) + '%' + ',KS_Cutoff= %s' % round(KS_cutoff, 2),fontsize=12)
    plt.legend(loc="upper left",fontsize=11)
    axes = plt.gca()
    axes.set_ylim(bottom=0)

# PR曲线
def plot_pr_fun(train_data,test_data,train_label,test_label, prob,target,dpi,bg_color,linewidth,save_name):
    '''
    :param train_data/test_data: 数据框
    :param train_label/test_label: 训练集和测试集模型标签
    :param prob: 预测的概率值
    :param target: 目标字段
    :param dpi:  图片像素
    :param bg_color:  背景色
    :param linewidth:  线宽
    :param save_name:  图片存储路径和名称
    :return:
    '''
    plt.figure(1, dpi=dpi)
    plt.grid(axis="y", alpha=0.6)  # 画y轴网格线
    # 设置figure窗体在颜色
    plt.rcParams['figure.facecolor'] = bg_color
    # 设置axes绘图区的颜色
    plt.rcParams['axes.facecolor'] = bg_color
    if len(test_data) > 10:
        precision, recall, thresholds = precision_recall_curve(train_data[target], train_data[prob])
        precision1, recall1, thresholds1 = precision_recall_curve(test_data[target], test_data[prob])
        plt.plot(recall, precision, label=train_label, linewidth=linewidth, color='#4F81BD')
        plt.plot(recall1, precision1, label=test_label, linewidth=linewidth, color='#8064A2')
        plt.title('Precision Recall Curve')
    else:
        precision, recall, thresholds = precision_recall_curve(train_data[target], train_data['prob'])
        plt.plot(recall, precision, label=train_label, linewidth=linewidth, color='#4F81BD')
        plt.title('Precision Recall Curve')
    plt.plot([0, 1], [0, 1], label='Break-Event Point', linewidth=linewidth, linestyle='--', color='#C0504D')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall Score')
    plt.ylabel('Precision Score')
    plt.legend(loc='best')
    plt.savefig(save_name)
    plt.close()
    #plt.clf()

# 黑白样本对应的模型得分分布情况
def black_white_dis_combine_plot(train_data,test_data,train_label,test_label,score,target,dpi,bg_color,bins,save_name):
    '''
    :param train_data:
    :param test_data:
    :param train_label:  训练集图片标题
    :param test_label:
    :param score:  模型分
    :param target:  目标字段
    :param dpi: 图片像素
    :param bg_color:  背景色
    :param bins:  直方图分箱数
    :param save_name:  保存图片名称
    :return:
    '''
    if len(test_data) > 10:
        plt.figure(1,figsize=(13, 5))
        plt.subplot(121)
        black_white_dis_plot(data=train_data, score=score, target=target, dpi=dpi, bg_color=bg_color, bins=bins,main=train_label)
        plt.subplot(122)
        black_white_dis_plot(data=test_data, score=score, target=target, dpi=dpi, bg_color=bg_color, bins=bins, main=test_label)
        plt.tight_layout()   ### 解决图片重叠问题
        plt.savefig(save_name)
        plt.close()
        #plt.clf()
    else:
        plt.figure(1,figsize=(7, 5))
        black_white_dis_plot(data=train_data, score=score, target=target, dpi=dpi, bg_color=bg_color, bins=bins, main=train_label)
        plt.savefig(save_name)
        plt.close()
        #plt.clf()        # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。

def black_white_dis_plot(data,score,target,dpi,bg_color,bins,main):
    '''
    :param data:
    :param score: 模型得分
    :param target: 目标字段
    :param figsize: 图片大小
    :param dpi:  图片像素
    :param bg_color: 背景
    :param bins: 分bin数
    :param mian:
    :param save_name:
    :return:
    '''
    plt.grid(axis="y",alpha=0.6)  # 画y轴网格线
    total = data.groupby(score)[target].count()
    plt.figure(1, dpi=dpi)
    # 设置figure窗体在颜色
    plt.rcParams['figure.facecolor'] = bg_color
    # 设置axes绘图区的颜色
    plt.rcParams['axes.facecolor'] = bg_color
    plt.hist(data[data[target] == 1][score], bins=bins, density=True, color='#1D364B', alpha=0.75, label='Bad_Score')
    plt.hist(data[data[target] == 0][score], bins=bins, density=True, color='#00FF00', alpha=0.55, label='Good_Score')
    plt.legend(loc="upper left")
    plt.xlabel('Score')
    plt.ylabel('%Obs')
    plt.title(main)

