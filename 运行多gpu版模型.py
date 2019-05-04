from collections import OrderedDict
import torch
def load_pretrainedmodel(modelname):
    pre_model = torch.load(modelname, map_location=lambda storage, loc: storage)["model"]
    #print(pre_model)
    if cuda:
        state_dict = OrderedDict()
        for k in pre_model.state_dict():
            name = k
            if name[:7] != 'module' and torch.cuda.device_count() > 1: # loaded model is single GPU but we will train it in multiple GPUS!
                name = 'module.' + name #add 'module'
            elif name[:7] == 'module' and torch.cuda.device_count() == 1: # loaded model is multiple GPUs but we will train it in single GPU!
                name = k[7:]# remove `module.`
            state_dict[name] = pre_model.state_dict()[k]
            #print(name)
        model_.load_state_dict(state_dict)
        #model_.load_state_dict(torch.load(modelname)['model'].state_dict())
    else:
        model_ = torch.load(modelname, map_location=lambda storage, loc: storage)["model"]
    return model_

def load_model(model):

    pre_model =torch.load(model,map_location='cpu')
    #print(pre_model)
    state_dict = OrderedDict()
    for k in pre_model:
        name = k
        print(name)
        if name[:6] == 'module':
            name = k[7:]  # remove `module.`
            print(name)
        state_dict[name] = pre_model[k]
    print(state_dict)
    return state_dict
if __name__ == '__main__':
    model_path = 'expr/crnn_Rec_done_9_51244.pth'
    load_model(model_path)