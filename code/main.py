import pandas as pd
from hierarchical import  HRP, KnowledgeTree
from utils import list_shares, Base
from sklearn.manifold import TSNE

rdf = Base.get_returns(list_shares)
tree = pd.read_excel('classification.xlsx', header=0)

# covariance and correlation matrcies for different time periods
cov_df_1, corr_df_1 = Base.get_stat(rdf, n = 60, start=0)
cov_df_2, corr_df_2 = Base.get_stat(rdf, n = 60, start=60)
cov_df_3, corr_df_3 = Base.get_stat(rdf, n = 60, start=120)
cov_df_4, corr_df_4 = Base.get_stat(rdf, n = 60, start=180)

cov_list = [cov_df_1, cov_df_2, cov_df_3, cov_df_4]
corr_list = [corr_df_1, corr_df_2, corr_df_3, corr_df_4]


weights_HRP_list = []
weights_tree_HRP_list= []
for cov_df, corr_df in zip(cov_list, corr_list):
    #ordinary HRP weights
    weights_HRP_list.append(HRP.getHRP(cov_df, corr_df))
    #tree HRP weights
    corr_new, link_new = KnowledgeTree.get_corr_hrp(tree, corr_df)
    weights_tree_HRP_list.append(HRP.getHRP(cov_df, corr_new, link_new))
    
# tsne
X_emb = TSNE(2).fit_transform(corr_new.values)
