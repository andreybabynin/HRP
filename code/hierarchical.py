import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.optimize import minimize
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import rankdata

#HRP class presents function from paper Building Diversified Portfolios that Outperform Out-of-Sample
class HRP:
    @classmethod
    def correlDist(cls, corr):
        # A distance matrix based on correlation, where 0<=d[i,j]<=1
        # This is a proper distance metric
        dist = ((1 - corr) / 2.)**.5  # distance matrix
        return dist
    
    
    @classmethod
    def getIVP(cls, cov, **kargs):
        # Compute the inverse-variance portfolio
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp
    @classmethod
    def getQuasiDiag(cls, link):
        # Sort clustered items by distance
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = sortIx.append(df0)  # item 2
            sortIx = sortIx.sort_index()  # re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    @classmethod
    def getClusterVar(cls, cov,cItems):
        # Compute variance per cluster
        cov_=cov.loc[cItems,cItems] # matrix slice
        w_=cls.getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar
    
    @classmethod
    def corr2cov(cls, corr,std):
        cov=corr*np.outer(std,std)
        return cov

    @classmethod
    def getRecBipart(cls, cov, sortIx):
        # Compute HRP alloc
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]  # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = cls.getClusterVar(cov, cItems0)
                cVar1 = cls.getClusterVar(cov, cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w


    @classmethod
    def getHRP(cls, cov, corr, link = None):
        # Construct a hierarchical portfolio
        if link is None: 
            dist = cls.correlDist(corr)
            link = sch.linkage(dist, 'single')
        if link is not None:
            cov = cls.corr2cov(corr, np.diag(cov)**.5)
            link = np.array([list(i) for i in link])
            
        # graph  
        '''
        dn = sch.dendrogram(link, labels=list_shares, leaf_rotation=90, distance_sort='descending',
                            count_sort= 'descendent')
        plt.show()
        '''
        sortIx = cls.getQuasiDiag(link)
        sortIx = corr.index[sortIx].tolist()
        hrp = cls.getRecBipart(cov, sortIx)
        return hrp.sort_index()
