
#####################################################################
#extracting data from youtube


import time
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from prediction import prediction,DistilBERTClass

import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
from selenium import webdriver
def scrap(link):
    data=[]
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    wd = webdriver.Chrome('chromedriver',options=chrome_options)
    wait = WebDriverWait(wd,5)
    wd.get(link)
    for item in range(20): 
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
        time.sleep(1)
        
    
    for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content"))):
        data.append(comment.text)
        if (len(data)==20):
            break
    wd.close()
    wd.quit()
    
    del data[0:5]
    #converting comments into dataframe
    import pandas as pd   
    df = pd.DataFrame(data, columns=['comment'])
    model = DistilBERTClass()
    
    return prediction(model,df)

if __name__=='__main__':
    scrap("https://www.youtube.com/watch?v=Wt-qu50f7Yo")
