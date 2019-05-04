#coding:utf-8
from PIL import Image
import lmdb
import io
import numpy as np
def read_from_lmdb(lmdb_path,img_save_to=None):
    try:
        # map_size = 3221225472
        data=dict()
        lmdb_env = lmdb.open(lmdb_path)
        with lmdb_env.begin() as lmdb_txn:
            lmbd_cursor = lmdb_txn.cursor()
            for key,value in lmbd_cursor:
                data[key] = value
                # tt = np.frombuffer(value,dtype='int8')
                # print(tt.shape[0])
                # print('value',value)
                # img=Image.frombytes('RGB',(280,32),value)
                # img.save(key+'.jpg')
        print(tuple(data.values())[0])
        s=tuple(data.values())[0]
        print(len(s))
        print ('s',s)
        img = Image.frombytes('RGB',(280,32),tuple(data.values())[0])
        img.show()
    finally:
        print('done')
if __name__ == '__main__':
    lmdb_path = './lmdb'
    img = Image.open('./train_images/20456343_4045240981.jpg')
    print(img.size)
    print(img.mode)
    print(len(np.array(img)))
    print(np.array(img).size)
    print(np.array(img).shape)
    read_from_lmdb(lmdb_path)