#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:31:25 2018

@author: Conory
"""
import requests
from bs4 import BeautifulSoup  #导入BeautifulSoup 模块
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import imutils
import threading 
from collections import defaultdict
from io import StringIO
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
import wx
from tkinter import *
from PIL import Image,ImageTk
import os  #导入os模块
import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

"""
若是通过anaconda安装的库
去到对应的环境：
/lib/python版本/xml/dom/minidom.py/注释掉1806行附近的函数writexml
中的		if encoding is None:
            writer.write('<?xml version="1.0" ?>'+newl)
        else:
            writer.write('<?xml version="1.0" encoding="%s"?>%s' % (
                encoding, newl))
才可以生成对能被代码识别的xml标注文件
"""

"""
对models/research/object_detection/utils/viasualization_utils.py文件进行修改
（1）导入库：
from xml.dom.minidom import Document
（2）修改函数visualize_boxes_and_labels_on_image_array
    1、增加两个传入参数
        fla=0,
        filename=None,
        file_name=None,
        fla用于是否需要生成xml文件
        file_name是xml的文件名
        filename是生成xml文件的绝对路径
    2、在return image：前增加语句：
        if flag==1 and fla==1:
        build_xml(image,box_to_color_map,box_to_display_str_map_1,filename)
（3）在py文件增加以下函数：
    def build_xml(image,box_to_color_map,box_to_display_str_map_1,filename):
        doc=Document()
        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)
        folder = doc.createElement('folder')
        xml_filename = doc.createElement('filename')
        path = doc.createElement('path')
        source = doc.createElement('source')
        size = doc.createElement('size')
        segmented = doc.createElement('segmented')
        annotation.appendChild(folder)
        annotation.appendChild(xml_filename)
        annotation.appendChild(path)
        annotation.appendChild(source)
        annotation.appendChild(size)
        annotation.appendChild(segmented)
        im_width=len(image[0])
        im_height=len(image)
        width=doc.createElement('width')
        height=doc.createElement('height')
        depth=doc.createElement('depth')
        width_text=doc.createTextNode(str(im_width))
        height_text=doc.createTextNode(str(im_height))
        depth_text=doc.createTextNode('3')
        width.appendChild(width_text)
        height.appendChild(height_text)
        depth.appendChild(depth_text)
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            object = doc.createElement('object')
            display_str_list=box_to_display_str_map_1[box]
            name=doc.createElement('name')
            name_text=doc.createTextNode(display_str_list[0])
            name.appendChild(name_text)
            object.appendChild(name)
            
            pose=doc.createElement('pose')
            pose_text=doc.createTextNode('Unspecified')
            pose.appendChild(pose_text)
            object.appendChild(pose)
            
            truncated=doc.createElement('truncated')
            truncated_text=doc.createTextNode('0')
            truncated.appendChild(truncated_text)
            object.appendChild(truncated)
            
            difficult=doc.createElement('difficult')
            difficult_text=doc.createTextNode('0')
            difficult.appendChild(difficult_text)
            object.appendChild(difficult)
            
            bndbox=doc.createElement('bndbox')
            xml_xmin=doc.createElement('xmin')
            xml_ymin=doc.createElement('ymin')
            xml_xmax=doc.createElement('xmax')
            xml_ymax=doc.createElement('ymax')
            xmin_text=doc.createTextNode(str(int(xmin*im_width)))
            xmax_text=doc.createTextNode(str(int(xmax*im_width)))
            ymin_text=doc.createTextNode(str(int(ymin*im_height)))
            ymax_text=doc.createTextNode(str(int(ymax*im_height)))
            xml_xmin.appendChild(xmin_text)
            xml_xmax.appendChild(xmax_text)
            xml_ymin.appendChild(ymin_text)
            xml_ymax.appendChild(ymax_text)
            bndbox.appendChild(xml_xmin)
            bndbox.appendChild(xml_xmax)
            bndbox.appendChild(xml_ymin)
            bndbox.appendChild(xml_ymax)
            object.appendChild(bndbox)
            annotation.appendChild(object)
        with open(filename, 'w') as f:
            f.write(doc.toprettyxml())
"""
# What model to download.
MODEL_NAME = '/Users/Conory/Downloads/ssd_inception_v2_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/猪脸识别.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/Users/Conory/Downloads/ssd_inception_v2_coco_2017_11_17', '猪脸识别.pbtxt')

NUM_CLASS=7

detection_graph=tf.Graph()
with detection_graph.as_default():
    od_graph_def=tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT,'rb')as fid:
        serialized_graph=fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def,name='')
label_map=label_map_util.load_labelmap(PATH_TO_LABELS)
categories=label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASS,use_display_name=True)
category_index=label_map_util.create_category_index(categories)

"""
def reg():
    cap=cv2.VideoCapture（0）
    with detection_graph.asdefault():
        with tf.Session(graph=detection_graph) as sess:
            writer =tf.summary.FileWriter("logs/",sess.graph)
            sess.run(tf.global_variables_init)
"""

"""
self.web_url是爬取的初始化网站，会根据网站的超链接一直爬取下去
修改调用pb，pbtxt的绝对路径
请注意修改生成xml的绝对路径。
保存爬取图片的绝对路径。

"""
class BeautifulPicture():
    def __init__(self):
        self.id=1
        self.headers= {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1'}
        #self.web_url=['https://unsplash.com']
        #self.web_url=['https://www.szu.edu.cn']
        #self.web_url=['https://www.sohu.com/a/115126578_498580']
        #self.web_url=['https://www.sohu.com/']
        self.img_url=[]
        self.web_url=['http://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gb18030&word=%B9%B7%CD%BC%C6%AC&fr=ala&ala=1&alatpl=adress&pos=0&hs=2&xthttps=000000']
        #self.web_url=['https://www.baidu.com/s?wd=%E7%8C%AA%E5%9B%BE%E7%89%87&rsv_spt=1&rsv_iqid=0xedb7263800037ac6&issp=1&f=8&rsv_bp=1&rsv_idx=2&ie=utf-8&rqlang=cn&tn=baiduhome_pg&rsv_enter=1&oq=%25E8%259E%25BA%25E4%25B8%259D%25E9%2592%2589%25E5%259B%25BE%25E7%2589%2587&rsv_t=d996pLpyWdWloBpgMdjOcAuIKdoVaYv28MbWMBQKsi%2FN0kq0dYN3A7Jp0aeNGvaE8ZS5&inputT=701&rsv_pq=f6b0b37a0003b146&rsv_sug3=96&rsv_sug1=66&rsv_sug7=100&bs=%E8%9E%BA%E4%B8%9D%E9%92%89%E5%9B%BE%E7%89%87']
        self.folder_path='/Users/Conory/spyder/picture'
    def reg(self,frame):
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                #writer =tf.summary.FileWriter("logs/",sess.graph)
                sess.run(tf.global_variables_initializer())
                frame=imutils.resize(frame,width=500)
                image_np=frame
                image_np_expanded=np.expand_dims(image_np,axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
                scores=detection_graph.get_tensor_by_name('detection_scores:0')
                classes=detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections=detection_graph.get_tensor_by_name('num_detections:0')
                
                (boxes,scores,classes,num_detections)=sess.run(
                        [boxes,scores,classes,num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                #flag=0
                #classes[0][i]=1,2,3,4,.........代表的是第几类
                """
                for i in range(90):
                    if scores[0][i]<0.5:
                        flag=i-1
                        break
                print(flag)
                if (flag>=0):
                    classes_1=classes[0][0:flag+1]
                    for i in range(len(classes_1)):
                        if classes_1[i]==1:
                            print("-----------------")
                """
                #if scores[0][0]>0.5 and classes[0][0]==1:    
                if scores[0][0]>0.5:
                    file_name=str(self.id)+'.jpg'
                    """
                    保存图片要放在画框之前，不然保存的图片会带有识别的框
                    """
                    cv2.imwrite("/Users/Conory/Desktop/spyder/images/"+file_name,frame)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        fla=1,
                        file_name=file_name,
                        filename='/Users/Conory/Desktop/spyder/annotations/xmls/'+str(self.id)+'.xml',
                        use_normalized_coordinates=True,
                        line_thickness=6)
                    print(file_name,'图片保存成功！')                  
                    cv2.imshow("",frame)
                    self.id+=1             
                    cv2.waitKey(1)
                else:
                    print("图片识别结果不符合")
    def scroll_down_one(self,driver):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        print("下滑网页，等待网页加载")
        time.sleep(10)
    def scroll_down(self,driver,times):
        for i in range(times):
            print("开始执行第", str(i + 1),"次下拉操作")
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            print("第", str(i + 1), "次下拉操作执行完毕")
            print("第", str(i + 1), "次等待网页加载......")
            time.sleep(20)
    def get_pic(self):
        driver=webdriver.PhantomJS('/Users/Conory/phantomjs/bin/phantomjs')
        for url in self.web_url:  
            #driver=webdriver.PhantomJS('/Users/Conory/phantomjs/bin/phantomjs')
            #driver=webdriver.Chrome('/Users/Conory/spyder/237')
            print(len(self.web_url))
            driver.get(url)
            print("success get from "+url)
            self.scroll_down_one(driver)
            #self.scroll_down(driver=driver,times=5)
            all_a=BeautifulSoup(driver.page_source,'lxml').find_all('a')
            for a in all_a:                           
                if a.has_attr('href'):
                    gg=a['href']
                    if gg not in self.web_url:
                        if 'http://' in gg :
                            self.web_url.append(gg)
                        elif 'https://' in gg :
                            self.web_url.append(gg)
            all_img=BeautifulSoup(driver.page_source,'lxml').find_all('img')
            #print("start to mkdir")
            #self.mkdir(self.folder_path)
            #os.chdir(self.folder_path)
            print("img标签的数量是：", len(all_img))
            tmps=[]
            for img in all_img:
                if img.has_attr('src')==False:
                    continue
                img_str=img['src']
                #print('a标签的style内容是：', img_str)
                if len(img_str)!=0:
                    if img_str not in self.img_url:
                        if 'https://' in img_str:
                            tmps.append(img_str)
                            self.img_url.append(img_str)
                        elif 'http://' in img_str:
                            tmps.append(img_str)
                            self.img_url.append(img_str)
                    """
                    else:
                        if img_str[0]=='/':
                            img_str=url+img_str
                            tmps.append(img_str)
                        else:
                            img_str=url+'/'+img_str
                            tmps.append(img_str)
                    """
            for tmp in tmps:
                #print(tmp)
                self.save_img(tmp)
        driver.close()
    """
    def mkdir(self,path):
        path=path.strip()
        isExists=os.path.exists(path)
        if not isExists:
            print('创建名字叫做', path, '的文件夹')
            os.makedirs(path)
            print('创建成功')
            return True
        else:
            print(path, '文件夹已经存在了，不再创建')
            return False
    """
    def save_img(self,url):
        print('开始请求图片地址，过程会有点长...')
        img=self.request(url)
        #file_name=str(self.id)+'.jpg'
        #f=open(file_name,'ab')
        image = np.asarray(bytearray(img.content))
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            return 
        w,h=image.shape[:2]
        if w>=50 and h>50:
            self.reg(image)
            #cv2.imshow("",image)
            #cv2.waitKey(1)
            #cv2.imwrite("%s.jpg"%self.id,image)
        else:
            print("图片尺寸太小")
        #f.write(img.content)
        #print(file_name,'图片保存成功！')
        #self.id+=1
        #f.close()
    def request(self,url):
        r=requests.get(url)
        return r
#driver=webdriver.Chrome('/Users/Conory/spyder/237')
#driver.get("http://www.baidu.com")
#print(driver.page_source)
#print(BeautifulSoup(driver.page_source,'lxml'))
#elem=driver.find_element_by_name("q")
#elem.clear()
#elem.send_keys("pycon")
#elem.send_keys(Keys.RETURN)
#assert "No results found." not in driver.page_source
#assert "Python" in driver.title
#driver.close()
beauty=BeautifulPicture()
beauty.get_pic()
"""
a=['123','234']
tmp='gg'
if tmp in a:
    print("gg")
else:
    a.append(tmp)
    print(a)
"""