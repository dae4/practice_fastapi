'''
This file is a barebones FastAPI example that:
	1. Accepts GET request, renders a HTML form at localhost:8000 allowing the user to
		 upload a image and select YOLO model, then submit that data via POST
	2. Accept POST request, run YOLO model on input image, return JSON output
Works with client.py
This script does not require any of the HTML templates in /templates or other code in this repo
and does not involve stuff like Bootstrap, Javascript, JQuery, etc.
'''


import base64
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from typing import List, Optional
from PIL import Image
from io import BytesIO
import torch
import random
import cv2
import base64
import numpy as np


app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]

@app.get("/")
def home(request: Request):
    '''
    Returns barebones HTML form allowing the user to select a file and model
    '''
    return templates.TemplateResponse('home.html', {
        "request": request,
      })

def results_to_json(results, model):
	''' Converts yolo model output to json (list of list of dicts)
	'''
	return [
				[
					{
					"class": int(pred[5]),
					"class_name": model.model.names[int(pred[5])],
					"bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
					"confidence": float(pred[4]),
					}
				for pred in result
				]
			for result in results.xyxy
			]

@app.post("/")
async def detect_via_web_form(request: Request,
							file_list: List[UploadFile] = File(...), 
							img_size: int = Form(640)):
  
  # model = torch.load('./model/yolov5s.pt')
  model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

  img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR) for file in file_list]


  results = model(img_batch.copy(), size = img_size)

  json_results = results_to_json(results,model)

  img_str_list = []
  #plot bboxes on the image
  for img, bbox_list in zip(img_batch, json_results):
    for bbox in bbox_list:
      label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
      plot_one_box(bbox['bbox'], img, label=label,color=colors[int(bbox['class'])], line_thickness=3)

  img_str_list.append(base64EncodeImage(img))

  #escape the apostrophes in the json string representation
  encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')

  return templates.TemplateResponse('show_results.html', {
  'request': request,
  'bbox_image_data_zipped': zip(img_str_list,json_results), #unzipped in jinja2 template
  'bbox_data_str': encoded_json_results,
  })


@app.post("/detect/")
async def detect_via_api(request: Request,
						file_list: List[UploadFile] = File(...), 
						img_size: Optional[int] = Form(640),
						download_image: Optional[bool] = Form(False)):
	
	'''
	Requires an image file upload, model name (ex. yolov5s). 
	Optional image size parameter (Default 640)
	Optional download_image parameter that includes base64 encoded image(s) with bbox's drawn in the json response
	
	Returns: JSON results of running YOLOv5 on the uploaded image. If download_image parameter is True, images with
			bboxes drawn are base64 encoded and returned inside the json response.
	Intended for API usage.
	'''


	model = torch.load('model/best.pt')

  #This is how you decode + process image with OpenCV

	img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR)
					for file in file_list]
	
	if download_image:
          #.copy() because the images are modified when running model, and we need originals when drawing bboxes later
          results = model(img_batch.copy(), size = img_size)
          json_results = results_to_json(results,model)

          for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
            for bbox in bbox_list:
              label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
              plot_one_box(bbox['bbox'], img, label=label, 
                  color=colors[int(bbox['class'])], line_thickness=3)

            payload = {'image_base64':base64EncodeImage(img)}
            json_results[idx].append(payload)

	else:
        #if we're not downloading the image with bboxes drawn on it, don't do img_batch.copy()
          results = model(img_batch.copy(), size = img_size)
          json_results = results_to_json(results,model)

	return json_results



def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
	# Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def base64EncodeImage(img):
	_, im_arr = cv2.imencode('.jpg', img)
	im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

	return im_b64


if __name__ == '__main__':
    import uvicorn
    
    app_str = 'server_custom:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)