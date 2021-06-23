import numpy as np
from PIL import Image
import cv2
import onnxruntime as nxrun
import time
import torchvision

def preprocess(frame):
    img = Image.fromarray(frame)
    model_image_size = (416, 416)
    
    size = tuple(reversed(model_image_size))
    iw, ih = img.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = img.resize((nw,nh), Image.BICUBIC)
    boxed_image = Image.new('RGB', size, (128,128,128))
    boxed_image.paste(image, ((w-nw)//2, (h-nh)//2))

    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def postprocess(prediction, conf_thres=0.25, iou_thres=0.7, classes=None, agnostic=True, multi_label=False, labels=(), max_det=300):
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def nms(dets, scores, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return np.array(keep)
    
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [np.zeros((0, 6))] * prediction.shape[0]
    #print(prediction)
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        conf = x[:, 5:].max(1, keepdims=True)
        j = np.argmax(x[:, 5:], 1).reshape(conf.shape)
        x = np.concatenate((box, conf, j), 1)[conf.all() > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        x = x[0]
        boxes, scores = x[:, :4], x[:, 4]
        i = nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    
    isDetect = np.array(output).any()

    if isDetect:
        return output
    else:
        return False

    
def draw_line(pred, input, img):
    def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

        coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
        coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
        coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
        coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])

        return coords
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(input.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(img, c1, c2, (128, 128, 128), thickness=tl, lineType=cv2.LINE_AA)
                #print(c1) # X Y
                #print(c2) # X+W Y+H
    bbox = (c1[0], c1[1], c2[0]-c1[0], c2[1]-c1[1])
    return img, bbox



sess = nxrun.InferenceSession("best.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

img = cv2.imread('tes.jpg')
print(img)
input = preprocess(img)
features = sess.run(None, {input_name: input})
print([a.shape for a in features])
pred = postprocess(features[0])
#print(pred)
img, bbox = draw_line(pred, input, img)
#print(bbox)
cv2.imshow('asdas', img)
cv2.waitKey()

'''
cap = cv2.VideoCapture('test.mp4')
tracker = cv2.legacy.TrackerMOSSE_create()

mode = None
detect = False
while True:
    ret, img = cap.read()
    
    if detect == False:
        input = preprocess(img)
        features = sess.run(None, {input_name: input})
        print([a.shape for a in features])
        pred = postprocess(features[0])
        if pred:
            img, bbox = draw_line(pred, input, img)
            tracker.init(img, bbox)
            detect = True
            mode = 'Predict - Success'
        else:
            detect = False
            mode = 'Predict - Failed'
    else:
        (success, box) = tracker.update(img)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 2)
            mode = 'Tracking - Success'
        else:
            mode = 'Tracking - Failed'
            detect = False

    cv2.imshow('asdas', img)
    print(mode)
    if cv2.waitKey(1) == ord('q'):
      break
cv2.destroyAllWindows()
'''
