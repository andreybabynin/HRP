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

# class to construct tree clustering from custom sources other than correlation and calculate linkage metrics
class KnowledgeTree:
    
    @classmethod
    def linkClusters(cls, lnk0,lnk1,items0,items1):
        # transform partial link1 (based on dist1) into global link0 (based on dist0)
        nAtoms=len(items0)-lnk0.shape[0]
        lnk_=lnk1.copy()
        for i in range(lnk_.shape[0]):
            i3=0
            for j in range(2):
                if lnk_[i,j]<len(items1):
                    lnk_[i,j]=items0.index(items1[int(lnk_[i,j])])
                else:
                    lnk_[i,j]+=-len(items1)+len(items0)
                # update number of items
                if lnk_[i,j]<nAtoms:
                    i3+=1
                else:
                    if lnk_[i,j]-nAtoms<lnk0.shape[0]:
                        i3+=lnk0[int(lnk_[i,j])-nAtoms,3]
                    else:
                        i3+=lnk_[int(lnk_[i,j])-len(items0),3]
            lnk_[i,3]=i3
        return lnk_
    
    @classmethod
    def updateDist(cls, dist0,lnk0,lnk_,items0,criterion=None):
        # expand dist0 to incorporate newly created clusters
        nAtoms=len(items0)-lnk0.shape[0]
        newItems=items0[-lnk_.shape[0]:]
        for i in range(lnk_.shape[0]):
            i0,i1=items0[int(lnk_[i,0])],items0[int(lnk_[i,1])]
            if criterion is None:
                if lnk_[i,0]<nAtoms:
                    w0=1.
                else:   
                    w0=lnk0[int(lnk_[i,0])-nAtoms,3]
                if lnk_[i,1]<nAtoms:
                    w1=1.
                else:
                    w1=lnk0[int(lnk_[i,1])-nAtoms,3]
                dist1=(dist0[i0]*w0+dist0[i1]*w1)/(w0+w1)
            else:
                dist1=criterion(dist0[[i0,i1]],axis=1) # linkage criterion
            dist0[newItems[i]]=dist1 # add column
            dist0.loc[newItems[i]]=dist1 # add row
            dist0.loc[newItems[i],newItems[i]]=0. # main diagonal
            dist0=dist0.drop([i0,i1],axis=0)
            dist0=dist0.drop([i0,i1],axis=1)
        return dist0

    @classmethod
    def getLinkage_hrp(cls, tree, corr):
        if len(np.unique(tree.iloc[:,-1]))>1:
            tree['All']=0 # add top level
        lnk0=np.empty(shape=(0,4))
        lvls=[[tree.columns[i-1],tree.columns[i]] for i in range(1,tree.shape[1])]
        dist0=((1-corr)/2.)**.5 # distance matrix
        items0=dist0.index.tolist() # map lnk0 to dist0
        for cols in lvls:
            grps=tree[cols].drop_duplicates(cols[0]).set_index(cols[0]).groupby(cols[1])
            
            for cat,items1 in grps:
                items1=items1.index.tolist()
                if len(items1)==1: # single item: rename
                    items0[items0.index(items1[0])]=cat
                    dist0=dist0.rename({items1[0]:cat},axis=0)
                    dist0=dist0.rename({items1[0]:cat},axis=1)
                    continue
                dist1=dist0.loc[items1,items1]
    
                lnk1=sch.linkage(ssd.squareform(dist1,force='tovector',
                                                checks=(not np.allclose(dist1,dist1.T))),
                                                optimal_ordering=True) # cluster that cat
                lnk_=cls.linkClusters(lnk0,lnk1,items0,items1)
                lnk0=np.append(lnk0,lnk_,axis=0)
                items0+=range(len(items0),len(items0)+len(lnk_))
                dist0=cls.updateDist(dist0,lnk0,lnk_,items0)
                # Rename last cluster for next level
                items0[-1]=cat
                dist0.columns=dist0.columns[:-1].tolist()+[cat]
                dist0.index=dist0.columns
        
        lnk0=np.array([tuple(i) for i in lnk0],dtype=[('i0',int),('i1',int), \
                      ('dist',float),('num',int)])    
        return lnk0

    @classmethod
    def getAtoms(cls, lnk,item):
        # get all atoms included in an item
        anc=[item]
        while True:
            item_=max(anc)
            if item_<=lnk.shape[0]:
                break
            else:
                anc.remove(item_)
                anc.append(lnk['i0'][item_-lnk.shape[0]-1])
                anc.append(lnk['i1'][item_-lnk.shape[0]-1])
        return anc
    @classmethod
    def link2corr(cls, lnk, lbls):
        # derive the correl matrix associated with a given linkage matrix
        corr=pd.DataFrame(np.eye(lnk.shape[0]+1),index=lbls,columns=lbls,
                          dtype=float)
        for i in range(lnk.shape[0]):
            x=cls.getAtoms(lnk,lnk['i0'][i])
            y=cls.getAtoms(lnk,lnk['i1'][i])
            corr.loc[lbls[x],lbls[y]]=1-2*lnk['dist'][i]**2 # off-diagonal values
            corr.loc[lbls[y],lbls[x]]=1-2*lnk['dist'][i]**2 # symmetry
        return corr
    
    @classmethod
    def get_corr_hrp(cls,tree, corr):
        link = cls.getLinkage_hrp(tree, corr)
        corr_ = cls.link2corr(link, corr.index)
        return corr_, link
