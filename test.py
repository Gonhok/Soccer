import torch
from scipy.optimize import linear_sum_assignment
from yolov3.models import *
from yolov3.utils.utils import *

def loadmodel(img_size=(1088, 1920)):
    cfg = './yolov3/cfg/yolov3-spp.cfg'
    weight = './yolov3/weights/yolov3-spp-ultralytics.pt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Darknet(cfg, img_size)
    model.load_state_dict((torch.load(weight, map_location=device))['model'])
    model.to(device).eval()

    return model

def predict_bbox(model, vframes, img_size=(1088, 1920), conf_thres=0.2, iou_thres=0.6):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vframes = vframes.permute(0, 3, 1, 2).float()
    vframes = vframes / 255.0
    vframes = torch.nn.functional.upsample_bilinear(vframes, size=img_size)
    vframes = vframes.to(device)

    preds = model(vframes, augment=False)[0]
    preds = preds.float()
    preds = non_max_suppression(preds, conf_thres, iou_thres, multi_label=False, classes=None, agnostic=False)

    preds = preds[0]
    preds[:, :4] = scale_coords(img_size, preds[:, :4], vframes.shape).round()
    preds = preds.detach().cpu().numpy()
    return preds

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = loadmodel()
print(1)
