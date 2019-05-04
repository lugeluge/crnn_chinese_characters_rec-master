#coding:utf-8
import  lmdb
import os
#如果 lmdb文件夹下没有data.mbd 或lock.mdb 文件 则会生成一个空的，如果有，不会覆盖
path = './testlmdb/'
if not os.path.exists(path):
    os.mkdir(path)
data  = lmdb.open(path)
txn = data.begin(write=True)
txn.put(str(0).encode(),str(00000).encode())
txn.put(key='1'.encode(),value='1abc'.encode())
txn.put(key='2'.encode(),value='2efg'.encode())
txn.put(key='3'.encode(),value='3efg'.encode())
txn.put(key='4'.encode(),value="4sdf".encode())
txn.put(key='5'.encode(),value='5df'.encode())

#删除
txn.delete(key='1'.encode())
#修改
txn.put(key='3'.encode(),value='3333'.encode())
#提交
txn.commit()
#关闭
data.close()
#开始
data = lmdb.open(path)
txn = data.begin()
print(txn.get(str(2).encode('utf-8')))
#查看编码方式
print(type(str(2).encode('utf-8')))
#遍历

for key ,value in txn.cursor():
    print(key,value)

#读取其他lmdb文件
env_db = lmdb.Environment('lmdb')
with env_db.begin() as txn:
    for key,value in txn.cursor():
        print(key,value)


