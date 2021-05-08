# Han Yu
# ID 1700012921

import requests
from bs4 import BeautifulSoup
import re
import time

OUTPUT=open("StatData.txt","w",encoding="utf-8",errors="ignore")


DOOR_LINK="http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2019/index.html"
BASE_URL=DOOR_LINK[0:DOOR_LINK.rfind("/")]


ITEM=["provincetr","citytr","countytr","towntr","villagetr"]
FIX=["","\t","\t\t","\t\t\t","\t\t\t\t"]


def Spider(oBS,rank,URL):
    
    allTr=oBS.find_all("tr",{"class":ITEM[rank]})
    
    for itemTR in allTr:
        
        if rank!=4:
            
            allLink=itemTR.find_all("a")
            l=len(allLink)
            i=0

            while i<l:
                
                if rank==0:
                    print("pro")
                    texti=str(allLink[i].get_text())
                    print(allLink[i].attrs["href"][0:2]+"0000000000"+texti,file=OUTPUT)
                    newURL=BASE_URL+"/"+allLink[i].attrs["href"]
                    i=i+1
                else:
                    
                    texti=str(allLink[i].get_text())+str(allLink[i+1].get_text())
                    print(FIX[rank]+texti,file=OUTPUT)
                    
                    newURL=URL[0:URL.rfind("/")]+"/"+allLink[i].attrs["href"]
                    i=i+2

                
                flag=True
    
                while flag:
                    try:
                        data1=requests.get(newURL)
                        data1.encoding="gb18030"
                        oBS1=BeautifulSoup(data1.text,"html.parser")

                        Spider(oBS1,rank+1,newURL)
                    except Exception:
                        time.sleep(1)
        
        else:
            allLink=itemTR.find_all("td")
            l=len(allLink)
            i=0

            while i<l:
                
                texti=str(allLink[i].get_text())+str(allLink[i+2].get_text())+str(allLink[i+1].get_text())
                print(FIX[rank]+texti,file=OUTPUT)
                
                i=i+3

def main():
    
    data=requests.get(DOOR_LINK)
    
    data.encoding="gb18030"
    
    html=data.text

    oBS=BeautifulSoup(html,"html.parser")
            
    Spider(oBS,0,BASE_URL)
            

main()

#eof