#爬虫 爬取股票数据https://q.stock.sohu.com/cn/002559/lshq.shtml
#https://q.stock.sohu.com/hisHq?code=cn_002559&start=20240326&end=20240724&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp&r=0.17691533997625297&0.011596120390038434=

import requests
import time
import pandas as pd
import json
root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"

number = "601857"

class CrawlerProcess():
        
    def requestData(self,number):
        session = requests.Session()
        session.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        url = "https://q.stock.sohu.com/hisHq?code=cn_"+number+"&start=20000326&end=20240724&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp&r=0.17691533997625297&0.011596120390038434="
        txt = ''   
        try:  
            r = session.get(url)
            txt = r.content.decode('gbk')
        except Exception as err:  
            try:  
                time.sleep(3)
                r = session.get(url)
                txt = r.content.decode('gbk')
            except Exception as err:  
                try:  
                    time.sleep(5)
                    r = session.get(url)
                    txt = r.content.decode('gbk')
                except Exception as err:
                    time.sleep(9)
                    r = session.get(url)
                    txt = r.content.decode('gbk')
        json_data  = txt[len("historySearchHandler")+2:-3]
        data = json.loads(json_data)
        hq = data.get("hq")
        # 将列表读取为DataFrame
        df = pd.DataFrame(hq)        
        # 设置列名
        df.columns = ['日期', '开盘','收盘','涨跌额','涨跌幅','最低','最高','成交量(手)','成交金额(万)','换手率']
        # 将字符串转换为日期时间
        df['日期'] = pd.to_datetime(df['日期'],format="%Y-%m-%d")
        
        # 格式化日期
        df['日期'] = df['日期'].dt.strftime('%Y/%m/%d')

        return df
    
    def loadstoreData(self,root,number):
        df = pd.read_csv(root+"data/"+number+"/"+number+".csv")  
        df['日期'] = pd.to_datetime(df['日期'],format="%m/%d/%Y")
        
        # 格式化日期
        df['日期'] = df['日期'].dt.strftime('%Y/%m/%d')
        return df
    
    def doProcess(self,root,number):
        stored = self.loadstoreData(root,number)
        loaded = self.requestData(number)
        new_store = pd.concat([stored, loaded], axis=0)
        
        
        new_store['日期'] = pd.to_datetime(new_store['日期'],format="%Y/%m/%d")  
        new_store['日期'] = new_store['日期'].dt.strftime('%Y/%m/%d')
        new_store = new_store.sort_values(by='日期', ascending=False)
        new_store.drop_duplicates(subset=['日期'],inplace=True)
        new_store.to_csv(root+"data/"+number+"/"+number+"22.csv", index=False)
        