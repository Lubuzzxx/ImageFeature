from fastapi import FastAPI, Request
import cv2
import numpy as np
app = FastAPI()
import base64

@app.get('/')
def root():

    return {"Message": 'This is my api'}



@app.get('/api/genhog')
async def readb64(data: Request):
    json = await data.json()
    image_str = json['img']
    encoded_data = image_str.split(',',1)
    img_str = encoded_data[1]
    decode = base64.b64decode(img_str)
    img = cv2.imdecode(np.frombuffer(decode, np.uint8),cv2.IMREAD_GRAYSCALE)
    s = (128,128)
    new_img = cv2.resize(img, s, interpolation=cv2.INTER_AREA)
    win_size =  new_img.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
    cell_size, num_bins)
    hog_descriptor = hog.compute(new_img)
    hog_descriptor_list = hog_descriptor.flatten().tolist()
    # print ('HOG Descriptor:', hog_descriptor)
    return {"HOG Vector": hog_descriptor_list}
    
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! I"}

@app.get("/api/genhog")
def read_str(item_str):
    img = readb64(item_str)
    return {"HOG": img}

