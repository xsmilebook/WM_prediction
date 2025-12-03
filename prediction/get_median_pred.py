import os
import numpy as np
from scipy import io as sio
from scipy import stats

def get_median_pred(project_folder, save_folder):
    """
    计算多次随机交叉验证中的中位数预测性能。
    对应get_medianPred.m的功能
    
    参数:
    -----------
    project_folder : str
        项目文件夹路径

    
    返回:
    --------
    corr_median : float
        所有运行中相关系数的中位数
    mae_median : float
        所有运行中平均绝对误差的中位数
    median_id : int
        中位数性能对应的运行ID
    """
    # MARK: 处理GGFC(灰质-灰质)
    fc_current = 'GGFC'
    folder_gg = os.path.join(project_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    save_folder_gg = os.path.join(save_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    if not os.path.exists(save_folder_gg):
        os.makedirs(save_folder_gg)
    # 初始化相关系数和MAE数组
    corr_overall_actual = np.zeros(101)
    mae_overall_actual = np.zeros(101)
    
    # 加载每次运行的结果
    for i in range(101):
        file_path = os.path.join(folder_gg, f'Time_{i}', 'Res_NFold.mat')
        mat_data = sio.loadmat(file_path)
        corr_overall_actual[i] = mat_data['Mean_Corr'][0][0]
        mae_overall_actual[i] = mat_data['Mean_MAE'][0][0]
    
    # 计算中位数值
    corr_median = np.median(corr_overall_actual)
    mae_median = np.median(mae_overall_actual)
    
    # 找出中位数相关系数对应的索引
    sorted_indices = np.argsort(corr_overall_actual)[::-1]  # 降序排序
    median_id = sorted_indices[50]  # 第51个项(从0开始索引)
    
    # 保存结果
    result_dict = {
        'Corr_Overall_Actual': corr_overall_actual,
        'MAE_Overall_Actual': mae_overall_actual
    }
    save_path = os.path.join(save_folder_gg, '2Fold_RandomCV_Corr_MAE_Actual_total.mat')
    sio.savemat(save_path, result_dict)

    # MARK: 处理GWFC
    fc_current = 'GWFC'
    folder_gw = os.path.join(project_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    save_folder_gw = os.path.join(save_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    if not os.path.exists(save_folder_gw):
        os.makedirs(save_folder_gw)
    # 初始化相关系数和MAE数组
    corr_overall_actual = np.zeros(101)
    mae_overall_actual = np.zeros(101)
    
    # 加载每次运行的结果
    for i in range(101):
        file_path = os.path.join(folder_gw, f'Time_{i}', 'Res_NFold.mat')
        mat_data = sio.loadmat(file_path)
        corr_overall_actual[i] = mat_data['Mean_Corr'][0][0]
        mae_overall_actual[i] = mat_data['Mean_MAE'][0][0]
    
    # 计算中位数值
    corr_median = np.median(corr_overall_actual)
    mae_median = np.median(mae_overall_actual)
    
    # 找出中位数相关系数对应的索引
    sorted_indices = np.argsort(corr_overall_actual)[::-1]  # 降序排序
    median_id = sorted_indices[50]  # 第51个项(从0开始索引)
    
    # 保存结果
    result_dict = {
        'Corr_Overall_Actual': corr_overall_actual,
        'MAE_Overall_Actual': mae_overall_actual
    }
    save_path = os.path.join(save_folder_gw, '2Fold_RandomCV_Corr_MAE_Actual_total.mat')
    sio.savemat(save_path, result_dict)
    
    # MARK: 处理WWFC
    fc_current = 'WWFC'
    folder_ww = os.path.join(project_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    save_folder_ww = os.path.join(save_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    if not os.path.exists(save_folder_ww):
        os.makedirs(save_folder_ww)
    # 初始化相关系数和MAE数组
    corr_overall_actual = np.zeros(101)
    mae_overall_actual = np.zeros(101)
    
    # 加载每次运行的结果
    for i in range(101):
        file_path = os.path.join(folder_ww, f'Time_{i}', 'Res_NFold.mat')
        mat_data = sio.loadmat(file_path)
        corr_overall_actual[i] = mat_data['Mean_Corr'][0][0]
        mae_overall_actual[i] = mat_data['Mean_MAE'][0][0]
    
    # 计算中位数值
    corr_median = np.median(corr_overall_actual)
    mae_median = np.median(mae_overall_actual)
    
    # 找出中位数相关系数对应的索引
    sorted_indices = np.argsort(corr_overall_actual)[::-1]  # 降序排序
    median_id = sorted_indices[50]  # 第51个项(从0开始索引)
    
    # 保存结果
    result_dict = {
        'Corr_Overall_Actual': corr_overall_actual,
        'MAE_Overall_Actual': mae_overall_actual
    }
    save_path = os.path.join(save_folder_ww, '2Fold_RandomCV_Corr_MAE_Actual_total.mat')
    sio.savemat(save_path, result_dict)

    return corr_median, mae_median, median_id

def partial_correlation(x, y, z):
    """
    计算控制变量z条件下x和y的偏相关系数
    使用公式: ρxy·z = (ρxy - ρxz·ρyz)/√[(1-ρxz²)(1-ρyz²)]
    
    参数:
    -----------
    x : array
        第一个变量
    y : array
        第二个变量
    z : array
        控制变量
    
    返回:
    --------
    r_xy_z : float
        偏相关系数
    p_value : float
        偏相关的p值
    """
    # 转换为numpy数组
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # 计算简单相关系数
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    
    # 使用公式计算偏相关系数
    r_xy_z = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    
    # 计算p值
    df = len(x) - 3  # 自由度，控制了一个变量，所以是n-3
    t = r_xy_z * np.sqrt(df) / np.sqrt(1 - r_xy_z**2)
    p_value = 2 * stats.t.sf(np.abs(t), df)
    
    return r_xy_z, p_value


def get_median_pred_partial_corr(project_folder, save_folder):
    """
    计算单个认知测量的中位数预测性能和偏相关。
    对应get_medianPred_partialCorr.m的功能的简化版本，仅处理单个目标。
    
    参数:
    -----------
    project_folder : str
        项目文件夹路径，包含要分析的认知测量
        
    返回:
    --------
    result_dict : dict
        包含该认知测量结果的字典
    """
    
    # Extract target string from the path for logging purposes
    target_str = os.path.basename(project_folder)
    print(f"处理 {target_str}")
    
    # 处理GGFC(灰质-灰质)
    fc_current = 'GGFC'
    plsr1_folder_gg = os.path.join(project_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    
    # 初始化数组
    corr_overall_actual = np.zeros(101)
    mae_overall_actual = np.zeros(101)
    
    # 加载每次运行的结果
    for i in range(101):
        file_path = os.path.join(plsr1_folder_gg, f'Time_{i}', 'Res_NFold.mat')
        mat_data = sio.loadmat(file_path)
        corr_overall_actual[i] = mat_data['Mean_Corr'][0][0]
        mae_overall_actual[i] = mat_data['Mean_MAE'][0][0]
    
    # 计算中位数相关系数
    corr_median_gg = np.median(corr_overall_actual)
    print(f"GGFC中位数相关系数为 {corr_median_gg}")
    
    # 计算中位数MAE
    mae_median = np.median(mae_overall_actual)
    
    # 找出中位数相关系数对应的索引
    sorted_indices = np.argsort(corr_overall_actual)[::-1]  # 降序排序
    median_id_gg = sorted_indices[50]  # 第51个项(从0开始索引)
    r_gg_total = corr_median_gg
    
    # 处理GWFC(灰质-白质)
    fc_current = 'GWFC'
    plsr1_folder_gw = os.path.join(project_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    
    # 初始化数组
    corr_overall_actual = np.zeros(101)
    mae_overall_actual = np.zeros(101)
    
    # 加载每次运行的结果
    for i in range(101):
        file_path = os.path.join(plsr1_folder_gw, f'Time_{i}', 'Res_NFold.mat')
        mat_data = sio.loadmat(file_path)
        corr_overall_actual[i] = mat_data['Mean_Corr'][0][0]
        mae_overall_actual[i] = mat_data['Mean_MAE'][0][0]
    
    # 计算中位数相关系数
    corr_median_gw = np.median(corr_overall_actual)
    print(f"GWFC中位数相关系数为 {corr_median_gw}")
    
    # 计算中位数MAE
    mae_median = np.median(mae_overall_actual)
    
    # 找出中位数相关系数对应的索引
    sorted_indices = np.argsort(corr_overall_actual)[::-1]  # 降序排序
    median_id_gw = sorted_indices[50]  # 第51个项(从0开始索引)
    r_gw_total = corr_median_gw
    
    # 处理WWFC(白质-白质)
    fc_current = 'WWFC'
    plsr1_folder_ww = os.path.join(project_folder, f'results_{fc_current}/PLSr1/AtlasLoading/RegressCovariates_RandomCV')
    
    # 初始化数组
    corr_overall_actual = np.zeros(101)
    mae_overall_actual = np.zeros(101)
    
    # 加载每次运行的结果
    for i in range(101):
        file_path = os.path.join(plsr1_folder_ww, f'Time_{i}', 'Res_NFold.mat')
        mat_data = sio.loadmat(file_path)
        corr_overall_actual[i] = mat_data['Mean_Corr'][0][0]
        mae_overall_actual[i] = mat_data['Mean_MAE'][0][0]
    
    # 计算中位数相关系数
    corr_median_ww = np.median(corr_overall_actual)
    print(f"WWFC中位数相关系数为 {corr_median_ww}")
    
    # 计算中位数MAE
    mae_median = np.median(mae_overall_actual)
    
    # 找出中位数相关系数对应的索引
    sorted_indices = np.argsort(corr_overall_actual)[::-1]  # 降序排序
    median_id_ww = sorted_indices[50]  # 第51个项(从0开始索引)
    r_ww_total = corr_median_ww
    
    # 获取fold分数以计算偏相关
    # 加载GG fold分数
    gg_fold0_path = os.path.join(plsr1_folder_gg, f'Time_{median_id_gg}', 'Fold_0_Score.mat')
    gg_fold1_path = os.path.join(plsr1_folder_gg, f'Time_{median_id_gg}', 'Fold_1_Score.mat')
    gg_fold0 = sio.loadmat(gg_fold0_path)
    gg_fold1 = sio.loadmat(gg_fold1_path)
    
    # 加载GW fold分数
    gw_fold0_path = os.path.join(plsr1_folder_gw, f'Time_{median_id_gw}', 'Fold_0_Score.mat')
    gw_fold1_path = os.path.join(plsr1_folder_gw, f'Time_{median_id_gw}', 'Fold_1_Score.mat')
    gw_fold0 = sio.loadmat(gw_fold0_path)
    gw_fold1 = sio.loadmat(gw_fold1_path)
    
    # 加载WW fold分数
    ww_fold0_path = os.path.join(plsr1_folder_ww, f'Time_{median_id_ww}', 'Fold_0_Score.mat')
    ww_fold1_path = os.path.join(plsr1_folder_ww, f'Time_{median_id_ww}', 'Fold_1_Score.mat')
    ww_fold0 = sio.loadmat(ww_fold0_path)
    ww_fold1 = sio.loadmat(ww_fold1_path)
    
    # 合并并排序GG分数
    gg_index_2fold = np.concatenate([gg_fold0['Index'].flatten(), gg_fold1['Index'].flatten()])
    gg_predict_score_2fold = np.concatenate([gg_fold0['Predict_Score'].flatten(), gg_fold1['Predict_Score'].flatten()])
    gg_test_score_2fold = np.concatenate([gg_fold0['Test_Score'].flatten(), gg_fold1['Test_Score'].flatten()])
    
    sort_indices = np.argsort(gg_index_2fold)
    gg_predict_score_2fold_sorted = gg_predict_score_2fold[sort_indices]
    
    # 合并并排序GW分数
    gw_index_2fold = np.concatenate([gw_fold0['Index'].flatten(), gw_fold1['Index'].flatten()])
    gw_predict_score_2fold = np.concatenate([gw_fold0['Predict_Score'].flatten(), gw_fold1['Predict_Score'].flatten()])
    gw_test_score_2fold = np.concatenate([gw_fold0['Test_Score'].flatten(), gw_fold1['Test_Score'].flatten()])
    
    sort_indices = np.argsort(gw_index_2fold)
    gw_predict_score_2fold_sorted = gw_predict_score_2fold[sort_indices]
    gw_test_score_2fold_sorted = gw_test_score_2fold[sort_indices].astype(float)
    
    # 合并并排序WW分数
    ww_index_2fold = np.concatenate([ww_fold0['Index'].flatten(), ww_fold1['Index'].flatten()])
    ww_predict_score_2fold = np.concatenate([ww_fold0['Predict_Score'].flatten(), ww_fold1['Predict_Score'].flatten()])
    ww_test_score_2fold = np.concatenate([ww_fold0['Test_Score'].flatten(), ww_fold1['Test_Score'].flatten()])
    
    sort_indices = np.argsort(ww_index_2fold)
    ww_predict_score_2fold_sorted = ww_predict_score_2fold[sort_indices]
    ww_test_score_2fold_sorted = ww_test_score_2fold[sort_indices].astype(float)
    
    # 计算偏相关
    partial_r_gw, p_gw = partial_correlation(
        gw_predict_score_2fold_sorted, 
        gw_test_score_2fold_sorted, 
        gg_predict_score_2fold_sorted
    )
    
    partial_r_ww, p_ww = partial_correlation(
        ww_predict_score_2fold_sorted, 
        ww_test_score_2fold_sorted, 
        gg_predict_score_2fold_sorted
    )
    
    print(f"GW偏相关系数: {partial_r_gw}, p={p_gw}")
    print(f"WW偏相关系数: {partial_r_ww}, p={p_ww}")
    
    # 保存结果
    result_dict = {
        'partialR_gw_total': partial_r_gw,
        'partialR_ww_total': partial_r_ww,
        'R_gg_total': r_gg_total,
        'R_gw_total': r_gw_total,
        'R_ww_total': r_ww_total,
        'medianID_GG': median_id_gg,
        'medianID_GW': median_id_gw,
        'medianID_WW': median_id_ww
    }
    save_path = os.path.join(save_folder, 'results_total.mat')
    sio.savemat(save_path, result_dict)
    
    return result_dict

def save_results_as_table(results, targetStr_list):
    """
    将结果保存为表格
    
    参数:
    -----------
    results : dict
        包含所有结果的字典
    targetStr_list : list
        目标字符串列表
    """
    # 初始化表头
    header = ['Target', 'R_gg_total', 'R_gw_total', 'R_ww_total', 'partialR_gw_total', 'partialR_ww_total']
    
    # 初始化表格
    table = [header]
    
    # 填充表格
    for targetStr in targetStr_list:
        result = results[targetStr]
        row = [
            targetStr, 
            f"{result['R_gg_total']:.4f}", 
            f"{result['R_gw_total']:.4f}", 
            f"{result['R_ww_total']:.4f}",
            f"{result['partialR_gw_total']:.4f}", 
            f"{result['partialR_ww_total']:.4f}"
        ]
        table.append(row)
        table.append(['', '', '', '', '', ''])
    
    # 保存表格
    save_path = '/ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/results/CCNP/results_table.csv'
    np.savetxt(save_path, table, fmt='%s', delimiter=',')




def main():
    """
    主函数，用于运行分析
    """
    # get_median_pred
    targetStr_list = ['FSIQ',  
                      'PerceptualReasoningIndex', 
                      'ProcessingSpeedIndex', 
                      'VerbalComprehensionIndex', 
                      'WorkingMemoryIndex']
    
    results = {}
    for targetStr in targetStr_list:
        project_folder = os.path.join('/ibmgpfs/cuizaixu_lab/zhaoshaoling/MSC_data/PNC/code/code_WMpost/step04_agePrediction/networkFC/CCNP/IQ', targetStr)
        save_folder = '/ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/results/CCNP/' + targetStr
        corr_median, mae_median, median_id = get_median_pred(project_folder, save_folder)
        print(f"中位数相关系数: {corr_median}, 中位数MAE: {mae_median}, 中位数ID: {median_id}")
        
        # get_median_pred_partial_corr
        result = get_median_pred_partial_corr(project_folder, save_folder)
        results[targetStr] = result

    save_results_as_table(results, targetStr_list)



if __name__ == "__main__":
    main()