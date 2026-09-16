import torch
from fusion_model_schemeB import create_fusion_model_schemeB

def main():
    m = create_fusion_model_schemeB(num_classes=3, dropout_p=0.5, pretrained=True)
    m.eval()
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    x3 = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        y = m(x1, x2, x3)
    print('SCHEME_B_FORWARD_OK', y.shape)

if __name__ == '__main__':
    main()