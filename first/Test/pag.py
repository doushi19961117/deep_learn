# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:10:53 2018

@author: Administrator
"""
# 获取网页中的html代码：
import requests
from bs4 import BeautifulSoup
import csv
def get_content(url, data=None):
    header = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.235'
    }
    while True:
        rep = requests.get(url, headers=header, timeout=1)
        rep.encoding = 'utf-8'
        break
    return rep.text
def get_data(html_text):
    final = []
    bs = BeautifulSoup(html_text, "html.parser")  # 创建BeautifulSoup对象
    body = bs.body  # 获取body部分
    print (body)
    data = body.find('div', {'id': '7d'})  # 找到id为7d的div
    ul = data.find('ul')  # 获取ul部分
    li = ul.find_all('li')  # 获取所有的li
    return final

#
# def write_data(data, name):
#     file_name = name
#     with open(file_name, 'a', errors='ignore', newline='') as f:
#         f_csv = csv.writer(f)
#         f_csv.writerows(data)


if __name__ == '__main__':
    url = 'http://www.weather.com.cn/weather15d/101020300.shtml'
    html = get_content(url)
    result = get_data(html)
    #write_data(result, 'weather.csv')
