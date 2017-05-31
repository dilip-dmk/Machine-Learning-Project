# -*- coding: utf-8 -*- 

import os, time, sys, shutil, codecs
reload(sys)
sys.setdefaultencoding('utf8')


def write_file(name, array):
  with codecs.open(name, "w", "utf-8") as f:
    for line in array:
      f.write(line + "\n")


if __name__ == '__main__':
  level1 = []
  level2 = []
  level3 = []
  level4 = []
  level5 = []
  with open('jdcomment-26w.txt') as f:
    for line in f:
      arr = line.split('\t')
      # 最后一项是评分
      if arr[-1][0] == '1':
        level1.append("".join(arr[:-1]))  
      if arr[-1][0] == '2':
        level2.append("".join(arr[:-1])) 
      if arr[-1][0] == '3':
        level3.append("".join(arr[:-1]))
      if arr[-1][0] == '4':
        level4.append("".join(arr[:-1]))
      if arr[-1][0] == '5':
        level5.append("".join(arr[:-1]))
  # 显示统计
  print "评分 1 记录数", len(level1)
  print "评分 2 记录数", len(level2)
  print "评分 3 记录数", len(level3)
  print "评分 4 记录数", len(level4)
  print "评分 5 记录数", len(level5)

  # 写入文件
  write_file("l1.txt", level1)
  write_file("l2.txt", level2)
  write_file("l3.txt", level3)
  write_file("l4.txt", level4)
  write_file("l5.txt", level5)
