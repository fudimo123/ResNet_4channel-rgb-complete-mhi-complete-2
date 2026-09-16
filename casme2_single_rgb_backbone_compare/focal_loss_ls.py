import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossLabelSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing support.
    
    Args:
        alpha (tensor, optional): Weights for each class.
        gamma (float, optional): Focusing parameter. Default: 2.
        reduction (str, optional): 'mean', 'sum' or 'none'. Default: 'mean'.
        smoothing (float, optional): Label smoothing factor (0.0 to 1.0). Default: 0.1.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean', smoothing=0.1):
        super(FocalLossLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        """
        inputs: (B, C) - Logits from the model
        targets: (B) - Integer class indices
        """
        num_classes = inputs.size(1)
        
        # 1. Compute Softmax Probabilities
        log_probs = F.log_softmax(inputs, dim=1) # log(p)
        probs = torch.exp(log_probs)             # p
        
        # 2. Construct Smoothed Targets
        # Create one-hot encoding
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # 3. Compute Focal Term: (1 - p_t)^gamma
        # pt is the probability of the true class (or smoothed target probability)
        # For Focal Loss, we typically focus on the probability of the target class.
        # With smoothing, we can use the probability of the *actual* ground truth class for the focal weight.
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1) # (B)
        focal_weight = (1 - pt).pow(self.gamma)
        
        # 4. Compute Cross Entropy with Smoothed Targets
        # CE = - sum(q * log(p))
        # We apply the focal weight to the entire CE loss for this sample
        loss = -torch.sum(true_dist * log_probs, dim=1) # (B)
        
        # 5. Apply Alpha Weighting (Class Balancing)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets] # (B)
            loss = loss * alpha_t
            
        # 6. Apply Focal Weight
        loss = loss * focal_weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
