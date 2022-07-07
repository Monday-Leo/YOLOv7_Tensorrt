import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from EfficientNMS import End2End

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov7.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--max-obj', type=int, default=100, help='topk')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='nms iou threshold')
    parser.add_argument('--score-thres', type=float, default=0.25, help='nms score threshold')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', default=True,help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device) # load FP32 model

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples
    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection
    model.eval()

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv) or isinstance(m, models.common.RepConv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    #print(model)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    model = End2End(model, max_obj=opt.max_obj, iou_thres=opt.iou_thres, score_thres=opt.score_thres, max_wh=False, device=device)

    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, 
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['num_dets','det_boxes','det_scores','det_classes'],
                          dynamic_axes= None)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        shapes = [opt.batch_size, 1, opt.batch_size, opt.max_obj, 4,
                opt.batch_size, opt.max_obj, opt.batch_size, opt.max_obj]
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))
        onnx.save(onnx_model, f)
        
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
