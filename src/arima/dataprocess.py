# -*- coding:utf-8 -*-
# '''
# Created on 2017年11月21日
# 
# @author: user
# '''
from math import radians, cos, sin, asin, sqrt  
from boto.dynamodb.condition import NULL
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import numpy as np
# from bokeh.models.axes import Axis

import statsmodels.api as sm


class Nile_predict:
    '''
全局变量

数据格式:
    df['ts','x','y','id']:[时间戳，经度，纬度，用户手机号]
    
    '''
    df = NULL
    #目标基站经纬度
    targetx,targety = 0,0
    #目标基站编号
    targetbs  = 0
    #目标基站活跃度序列S[]
    targetAbs = []
    #目标基站人数序列[]
    targetmbs = []
    #基站列表[]
    l_bs = []
    #与目标基站所在簇所含的基站（不包含目标基站），调用select()后更新为对目标基站吸引力最大的m个基站
    cluter = []
    #｛基站:人数序列[]｝
    matix = {}
    #时间标签S[]
    timelabel = []
    #{基站:{时间段:S[id]id的series列表}}：
    idstream = {}
    #｛bs:[有向交互人流的时间序列]｝
    vecst = {}
    '''
        方法
    '''


    '''
                初始化参数：目标基站经度，纬度
    '''

    def __init__(self,df,x=113.067825,y=113.067825):
        self.targetx = x
        self.targety = y
        self.df = df
        pass
    
    '''
        添加bs列唯一标识基站
        
    '''
    def addbs(self):
       
        bs = []
        #时间戳转换datatime
        
        #将时间戳中的秒部分舍弃
        self.df['ts'] = self.df['ts']//60000*60
        self.df['ts'] = pd.to_datetime(self.df['ts'],unit='s')
    #     print df['ts']
        #选取经纬度坐标，去重后的索引作为基站id
        xy = self.df[['x','y']].drop_duplicates(['x','y'])
    #     print len(xy)
        for i in range(self.df.shape[0]):
            row = self.df[['x','y']].ix[i]
            bs.append(xy[(xy.x==row[0])&(xy.y==row[1])].index[0])
        self.df['bs1'] = pd.Series(np.array(bs))
    #     print df['bs1'].drop_duplicates().shape
        #记录目标基站编号
        self.targetbs = xy[(xy.x==self.targetx)&(xy.y==self.targety)].index[0]
#         print ('self.targetbs is %s'%(self.targetbs))
#         return df
        pass
    
    '''
        参数：width:时间跨度,即使用多少个时间段;返回基站x人数序列矩阵｛bs1:mbs｝的dataframe格式
    '''
    def split(self,df,width = 50):
        
#         print ('start split')
        #得到时间标签,排序并按步长截取
        timelabel = df['ts'].drop_duplicates().tolist()
        timelabel.sort()
#         print len(timelabel)
        timelabel = timelabel[-1:-width-1:-1]
#         print len(timelabel)
        #得到基站列表
        l_bs = df.bs1.drop_duplicates()
    #     print l_bs.shape
        for bs in l_bs:
            
            #检查bs是否具有timelabel中要求的时间点集
            flag = 1
            timelist_bs = list(df[(df.bs1==bs)]['ts'].drop_duplicates())
#             print (timelist_bs)
            for i in timelabel:
                if i not in timelist_bs:
                    flag = 0
            if flag ==0:
                continue
            #
            self.matix[bs] = []        
            #对每个基站得到｛ti:df[pid]｝
            mbs=dict(list(df[(df.bs1==bs)][['id','ts']].groupby('ts')))
    #         print mbs
            #对时间序列排序
#             l_t = list(mbs.keys())
    #         print ('基站%d时间点数量为: %d'%(bs,len(l_t)))
    #         l_t = mbs.keys().sort()
            #存放id人数
            npid = []
            #存放id
            ids = {}
            
            for i in range(len(timelabel)):            
                #得到各时段pid序列并去重，为df
    #             print ('现在是基站 %d'%(bs))
    #             print ('现在是第 %d次'%(i))
    #             for j in range(i,i+length):                
    #                 l_pid = mbs[l_t[j]]['id'].append(mbs[l_t[j+1]]['id'],verify_integrity=True).drop_duplicates()
                #记录序列
                ids[timelabel[i]] = mbs[timelabel[i]]['id']
    #             print ids[l_t[i]].shape[0]
                
    #             print ('\n\n基站%d在%s的人数序列为：'%(bs,l_t[i]))
    #             print ids[l_t[i]]
                #统计人数
                npid.append(ids[timelabel[i]].shape[0])
    #         print npid
            #汇总各基站id序列           
            self.idstream[bs] = ids
            #截取取给定时间段内人数放进矩阵
#             npid = npid[-1:-width-1:-1]#测试完记得还原[-1:-width:-1]
            self.matix[bs] = npid
            
            
#         print self.matix
#         print list(self.matix.keys())
        #标记目标基站人数序列
        
        self.targetmbs = self.matix[self.targetbs]
        #返回基站x人数矩阵
#         print pd.DataFrame(self.matix).T
#         return pd.DataFrame(self.matix).fillna(0).T
        df = pd.DataFrame(self.matix).T

        self.l_bs = df.index.tolist()
        return df
        pass
    
    '''
        接受 基站x人数矩阵，获得并返回基站x活跃度序列
    '''
    def Activate(self,df,threshold):
        
        #做差分,取绝对值
        
        Abs = np.abs(df.diff(axis=1).dropna(axis = 1,how = 'all'))
#         print Abs
        cAbs = Abs.copy(deep=True)
        #获得基站活跃度序列
        for i in range(Abs.shape[0]):
            cAbs.iloc[i] = Abs.iloc[i].map(lambda x: [0,1] [x>=threshold])
#             print cAbs.iloc[i]
        #标记目标基站活跃度序列
#         print cAbs
        self.targetAbs = cAbs.T[self.targetbs]
        return cAbs
        pass
    
    '''
        #接受基站x活跃度序列，得到与目标基站一类的基站cluter[]
        #训练模型，找最优聚类簇数量
    '''
    def kmean(self,df,m=10):
        
        kmeans = 0
        X = df.as_matrix() 
#         X = df
#         print df
        
        silhouette_score = 0
        for i in range(2,m+1):
            kmean = KMeans(n_clusters=i,init='k-means++',max_iter=500)
            kmeans_model = kmean.fit(X)
#             print (kmeans_model.labels_)
            ss = metrics.silhouette_score(X,kmeans_model.labels_,metric='euclidean')
#             print ss
            if ss > silhouette_score :
                silhouette_score = ss
                kmeans = kmeans_model
            
        
        
        labels = kmeans.labels_
        self.targetAbs = df.iloc[:1]#
        tlabel = kmeans.predict([self.targetAbs])
#         print tlabel
        for i in range(len(self.l_bs)):
            if labels[i]==tlabel and labels[i]!=self.targetbs:
                self.cluter.append(self.l_bs[i])
#         self.cluter.remove(self.targetbs)
#         print self.cluter
        pass
    
    """ 
        Calculate the great circle distance between two points  
        on the earth (specified in decimal degrees) 
        #通过经纬度计算距离的函数
    """ 
    def haversine(self,lon1, lat1, lon2=targetx, lat2=targety): # 经度1，纬度1，经度2，纬度2 （十进制度数）  
         
        # 将十进制度数转化为弧度  
        lon1= map(radians, np.array([lon1]))  
        lat1= map(radians, np.array([lat1]))
        lon2= map(radians, np.array([lon2]))
        lat2= map(radians, np.array([lat2]))
        lon1 = np.array(list(lon1)).reshape(-1,1)
        lon2 = np.array(list(lon2)).reshape(-1,1)
        lat1 = np.array(list(lat1)).reshape(-1,1)
        lat2 = np.array(list(lat2)).reshape(-1,1)
        # haversine公式  
        dlon = lon2 - lon1
        dlat = lat2 - lat1 
    
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2  
        c = 2 * np.arcsin(np.sqrt(a))   
        r = 6371 # 地球平均半径，单位为公里  
        return c * r * 1000  
        pass
    
    '''
        #计算基站间吸引力强度
                接受参数：bs1:基站bs1的人数的时间序列，x,y:基站bs1经纬度
        #model = 0:计算并返回吸引力序列向量的模
        #model = 1:计算并返回吸引力序列向量
    '''
    def G(self,bs1,x,y,model=0):
        
        #得到基站bs1与目标基站的距离
        l = self.haversine(x, y)
        
        if model ==0:
            sum =0
            
            for i in range(len(bs1)):
                sum+=(bs1[i]*self.targetmbs[i])**2
                print sum
            return sqrt(sum)/l**2
        
        
        if model ==1:
            glist = []
            for i in bs1:
                glist.append(self.targetmbs[i]*bs1[i]/l**2)
            return glist
        pass
    
    '''
        #接受基站x人数矩阵，更新cluter为与目标基站吸引力最强的m个基站
    '''
    def select(self,m=8):
        
        l = {}
        max_m = []
        for i in self.cluter:
            
            g = self.G(self.matix[i],self.df[self.df.bs1==i]['x'].iloc[0],self.df[self.df.bs1==i]['y'].iloc[0])
            g = g[0][0]
            
            l[g] = i
        listg = l.keys()
        listg.sort()
#         print listg
        for i in listg[-1:-m:-1]:
            max_m.append(l[i])
            
        self.cluter = max_m
        pass
    
    '''
        #接受原始dataframe（增加bs标签之后的）,得到并返回簇内其他基站对目标基站对吸引力序列 {bs:[tbs]}
    '''
    def getTbs(self):
        
        tbs = {}
        for i in self.cluter:
            tbs[i] = self.G(self.matix[i],self.df[self.df.bs1==i]['x'],self.df[self.df.bs1==i]['y'],model = 1)
            pass
        return tbs
        pass
    
    '''
    #得到cluter中基站人数均值序列[]
    '''    
    def getavg(self):
        '''
        #得到cluter中基站人数均值序列[]
        '''
#         lavg = []
        
        m_avg = pd.DataFrame(self.matix)[self.cluter].T.mean().tolist()
        return m_avg
        pass
    
    # def liner(X):
    #     
    #     pass
    
        '''
        #计算有向人流
        #返回｛bs:[有向交互人流的时间序列]｝
        '''    
    def vecstream(self):
        '''
        #计算有向人流
        #返回｛bs:[有向交互人流的时间序列]｝
        '''
        #｛bs:[有向交互人流的时间序列]｝
        vecst = {}
        for bs in self.cluter:
            vecst[bs] = []
            n_in = 0
            n_out =0
            for i in range(len(self.timelabel)-1):
                n_in = len(self.idstream[bs][self.timelabel[i]].intersection (self.idstream[self.targetbs][self.timelabel[i+1]]))
                n_out = len(self.idstream[bs][self.timelabel[i+1]].intersection (self.idstream[self.targetbs][self.timelabel[i]]))
            vecst[bs].append(n_in-n_out)
        self.vecst = vecst
    #     return vecst
    
    
    '''
                        接受序列和预测时间点（时间戳类型），返回预测值npredict数量的预测值
    ''' 
#     def arima(self,series,npredict=1):
#         
#         date = pd.Series(series,dtype=np.float)
#         #为序列添加日期索引
#         date_index = pd.date_range('2010', periods=len(series)+npredict, freq='D')
#         date.index = date_index[:-npredict]
# #         p = pd.to_datetime(predict)
# #         tl = pd.to_datetime(self.timelabel)
# #         series.index = tl
#         arma_mod80 = sm.tsa.ARMA(series,(5,0)).fit()
#         predict_dta = arma_mod80.predict(date.index[len(series)], date.index[len(series)+npredict-1], dynamic=True)
#         print(predict_dta)
#         return predict_dta
#         pass
    def lstm(self,list):
        
        pass
    
    '''
    预测函数，对cluter中基站bs对目标基站的有向人流数vecst序列[]和目标基站的人数targetmbs调用arima()作时间序列预测，
    返回目标基站人数预测值
    '''
    def predict(self):
        pre_data = []
        for bs in self.cluter:
            pre_data.append(self.lstm(self.vecst[bs]))
            pass
        return sum(pre_data)+self.lstm(self.targetmbs)
        pass
    
def gettestdata(list):
    return pd.DataFrame(np.random.randint(2,size=(len(list),20) ),index =list)
           
    pass
    '''
    测试函数
    '''
def tmain():
    df = pd.read_csv('data.csv')
    n = Nile_predict(df)
#     df['ts'] = pd.to_datetime(df['ts'])
#     print df['ts']
    n.addbs()
    n.split(df)
#     mbs_matix = n.split(df)
#     n.Activate(mbs_matix,5)
    test_matix = gettestdata(n.l_bs)
    n.kmean(test_matix)
#     n.kmean(n.Activate(mbs_matix,5))
#     
    n.select()
    n.vecstream()
    n.predict()
    
#     #吸引力序列
#     tbs = getTbs(df)
#     lavg = getavg()
        
    pass

# def main():
#     df = pd.read_csv('tpd.csv',header=0)
# #     print (dict(list(df[df.bs==1].groupby('id')['ts'])))#返回各组别的字典
# #     prt (df[df.bs==1][['ts','value']].groupby('ts').count())
# #     print (df.id.drop_duplicates())
# #     print df.as_matrix()
# #     addbs(df)
# #     print df['id'].drop_duplicates()
# #     print df[(df.id==33)].index
# #     print df.diff(1,axis = 0)
# #     print df.groupby('id')
#     for (x,y) in df.groupby('id'):
#         print x,y
#         break

#     print df.ix[2].map(lambda x: [0,1] [x>10])#lambda x:[flase,true][if excption]
#     print (df.sort(["id"],ascending=False))

if __name__ =='__main__':
    tmain()