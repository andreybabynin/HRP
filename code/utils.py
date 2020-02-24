import pandas as pd
import numpy as np

#this class is used to upload and aggregate data
class Base:
    @staticmethod
    def get_returns(list_shares):
        comb_df = pd.DataFrame()
        for n, i in enumerate(list_shares):
            file = list_shares[n]+'.ME'+'.csv'
            df = pd.read_csv(file, header = 0, usecols = ['Date', 'Adj Close'])
            df.index = df.Date
            df.drop(columns = 'Date', inplace = True)
            df.rename(columns ={'Adj Close': i}, inplace=True)
            if len(comb_df) == 0:
                comb_df = df
            else:
                comb_df = pd.merge(comb_df, df, how = 'left', left_index=True, right_index=True)
            
        comb_df_ret = comb_df.pct_change(1).iloc[1:]
        return comb_df_ret

    @staticmethod
    def get_stat(returns, start = 0, n=-1):
        if n==-1:
            cov = np.cov(returns.values, rowvar=False)
            corr = np.corrcoef(returns.values, rowvar=False)
        else:
            cov = np.cov(returns.iloc[start:min(start+n, returns.shape[0]-1)].values, rowvar=False)
            corr = np.corrcoef(returns.iloc[start:min(start+n, returns.shape[0]-1)].values, rowvar=False)
                
        # form DataFrame
        corr_df = pd.DataFrame(corr)
        cov_df = pd.DataFrame(cov)  
        
        # add titles
        corr_df.index = list_shares
        corr_df.columns = list_shares
        cov_df.index = list_shares
        cov_df.columns = list_shares
        return cov_df, corr_df
