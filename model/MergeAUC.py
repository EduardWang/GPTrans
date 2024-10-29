import pandas as pd
from sklearn import metrics
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # pre_esm = np.lib.format.open_memmap('../features_npy/esm/FULL/pre/161/FULL_esm.npy')
    # pre_prot = np.lib.format.open_memmap('../features_npy/prottrans/FULL/pre/161/FULL_prot.npy')
    # pre_label = np.lib.format.open_memmap('../features_npy/labels/FULL/pre/161/FULL_labels.npy')

    file_path = 'E:\DataSets\GPCR\source\\3_mo.xlsx'  # 请替换为您的文件路径
    sheet_name = 'Sheet1'
    df = pd.read_excel(file_path,sheet_name)
    cssl = 'ClinSigSimple'
    amp = 'AlphaMissense'
    y_true = df[cssl]
    y_pred_am = df[amp]
    y_pred_am = y_pred_am.replace('ambiguous', 0)
    fpr_am, tpr_am, thresholds_am = metrics.roc_curve(y_true, y_pred_am, pos_label=1)
    roc_auc_am = metrics.auc(fpr_am, tpr_am)


    esm1bp = 'ESM1b'
    # 将某列数据输入到一个numpy数组中
    y_pred_1b = df[esm1bp].values
    fpr_1b, tpr_1b, thresholds_1b = metrics.roc_curve(y_true, y_pred_1b, pos_label=1)
    roc_auc_1b = metrics.auc(fpr_1b, tpr_1b)


    pvp = 'PROVEAN'
    y_pred_pvp = df[pvp].values
    fpr_pvp, tpr_pvp, thresholds_mp = metrics.roc_curve(y_true, y_pred_pvp,pos_label=1)
    roc_auc_pv = metrics.auc(fpr_pvp, tpr_pvp)

    pp2p = 'PolyPhen-2'
    # 将某列数据输入到一个numpy数组中
    y_pred_pp2p = df[pp2p]
    y_pred_pp2p = y_pred_pp2p.replace('ERROR', 0)
    y_pred_pp2p = y_pred_pp2p.fillna(0).replace('', 0)
    fpr_pp2p, tpr_pp2p, thresholds_pp2p = metrics.roc_curve(y_true, y_pred_pp2p, pos_label=1)
    roc_auc_pp2 = metrics.auc(fpr_pp2p, tpr_pp2p)

    siftp = 'SIFT'
    # 将某列数据输入到一个numpy数组中
    y_pred_siftp = df[siftp]
    y_pred_siftp = y_pred_siftp.replace('ERROR', 0)
    y_pred_siftp = y_pred_siftp.fillna(0).replace('', 0)
    fpr_siftp, tpr_siftp, thresholds_siftp = metrics.roc_curve(y_true, y_pred_siftp, pos_label=1)
    roc_auc_sift = metrics.auc(fpr_siftp, tpr_siftp)


    # 读取xlsx文件
    file_path = 'E:\Projects\GPTrans\\result\muthtp\\test\m\\fcv_181_test_1024_add_label_1_pred_mcc.xlsx'
    sheet_name='Sheet2'
    df2 = pd.read_excel(file_path,sheet_name)
    cn13 = 'True'
    cn14 = 'Predict'
    # 将某列数据输入到一个numpy数组中
    y_true3_final = df2[cn13].values
    y_pred3_final = df2[cn14].values

    fpr_gpt, tpr_gpt, thresholds_gpt = metrics.roc_curve(y_true3_final, y_pred3_final, pos_label=1)
    roc_auc_gpt = metrics.auc(fpr_gpt, tpr_gpt)



    #使用模型预测
    # y_true=pre_label
    # full_model = get_model()
    # full_model.load_weights('../save_model/fullindep/full/model_regular.h5')
    # y_pred_full= full_model.predict([pre_esm, pre_prot]).reshape(-1, )
    # fpr_full, tpr_full, thresholds = metrics.roc_curve(y_true, y_pred_full,pos_label=1)
    # roc_auc_full = metrics.auc(fpr_full, tpr_full)


    # 绘制ROC曲线图
    plt.figure(figsize=(9, 8))
    plt.rcParams['font.family'] = 'Arial'

    # plt.plot(fpr_final, tpr_final, color='red', label='MetalTrans ROC curve (AUC = %0.3f)' % roc_auc_final)
    # plt.plot(fpr2_final, tpr2_final, color='orange', label='$\mathregular{MetalTrans_{Each}}$ ROC curve (AUC = %0.3f)' % roc_auc2_final)
    # plt.plot(fpr_mp, tpr_mp, color='deepskyblue', label='$\mathregular{MetalPrognosis_{Final}}$ ROC curve (AUC = %0.3f)' % roc_auc_mp)

    plt.plot(fpr_gpt, tpr_gpt, color='red', label='GPTrans ROC curve (AUC = %0.3f)' % roc_auc_gpt)
    plt.plot(fpr_am, tpr_am, color='#FEA040', label='AlphaMissense ROC curve (AUC = %0.3f)' % roc_auc_am)
    plt.plot(fpr_pp2p, tpr_pp2p, color='gold', label='PolyPhen-2 ROC curve (AUC = %0.3f)' % roc_auc_pp2)
    plt.plot(fpr_1b, tpr_1b, color='#8DECF5', label='ESM 1b ROC curve (AUC = %0.3f)' % roc_auc_1b)
    plt.plot(fpr_siftp, tpr_siftp, color='deepskyblue', label='SIFT ROC curve (AUC = %0.3f)' % roc_auc_sift)
    plt.plot(fpr_pvp, tpr_pvp, color='dodgerblue', label='PROVEAN ROC curve (AUC = %0.3f)' % roc_auc_pv)


    plt.plot([0, 1], [0, 1], 'darkgray', linestyle='--')
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel('False Positive Rate',fontsize=19)
    plt.ylabel('True Positive Rate',fontsize=19)
    plt.title('Receiver Operating Characteristic Curves',fontsize=20)
    plt.legend(loc='lower right',fontsize=13.5)
    # plt.legend(frameon=False, loc='lower right')
    # plt.savefig('../pic/roma/MTP_MergeCurve.png', dpi=300)
    plt.savefig('pic/final/MutHTPM_MergeCurve.png', dpi=300)
    plt.show()
