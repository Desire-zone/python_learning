from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import random
base_url="https://baike.baidu.com"
his=["/item/%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB/5162711?fr=aladdin"]
for i in range(20):
    url=base_url+his[-1]
    html=urlopen(url).read().decode('utf-8')
    soup=BeautifulSoup(html,features='lxml')
    print(i+1,soup.find('h1').get_text(), '    url: ',his[-1])
    sub_urls=soup.find_all("a",{"target":"_blank",
                                "href":re.compile("/item/(%.{2})+$")})
    #print(sub_urls)
    if len(sub_urls) !=0:
        his.append(random.sample(sub_urls,1)[0]['href'])
    else:
        his.pop()#默认去掉最后一个元素即-1
    #print(his)