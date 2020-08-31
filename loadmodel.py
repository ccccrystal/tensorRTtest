from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))