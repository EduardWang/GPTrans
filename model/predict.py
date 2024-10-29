import test_indep
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATASET='muthtp'

if __name__ == "__main__":
    excel_path=f'result/final/{DATASET}/all/all_GPTrans.xlsx' 
    sheet_name='Sheet2'
    df=pd.read_excel(excel_path,sheet_name)
    pl=df['Predict']
    tl=df['True']
    sf=f'result/{DATASET}_GPTranspred.xlsx'
    test_indep.fcvtest(pl,tl,sf)