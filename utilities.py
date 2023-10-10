# -*- coding: utf-8 -*-
"""

"""
def prepare_graphs_labels(conMat, label, feats, graphThreshold, SpDisAdj):
    # This function prepares graphs and their corresponding labels for training and testing
    # list of dgl graphs
    graphs = []
    # list of labels for graphs
    labels = []
    chan_names = ["1", "2", "3", "4", "5", "6",
                  "7", "8", "9", "10", "11", "12",
                  "13", "14", "15", "16", "17", "18"]
    event_id   = { 1: "seizure", 0: "non_seizure"}
    sfreq      = 32
    n_channels = 18

    # create a progress bar
    bar = Bar('Thresholding graphs and preparaing labels', max = 39, check_tty = False, hide_cursor = False)
    G_threshold = graphThreshold
    SpDis = np.load("/mygoogledrive/MyDrive/Spatial_Distance_Adjacency_plusglobal.npy").astype(int)

    for i in range(39):        
        # thresholding the graphs
        tempSubject = feats[i]
        tempFeats   = feats[i]
        # a loop over seizure and non-seizre segments of each subject 
        for j in range(tempFeats.shape[0]):       
            # extracts each connectivity matrix
            if SpDisAdj:
              unweightedMat = SpDis
            else:
              unweightedMat = (tempConMat >= G_threshold).astype(int)
            # converts each connectivity matrix to a scipy sparse matrix
            tempGraph = scipy.sparse.coo_matrix(unweightedMat)           
            # converts each scipy sparse matrix to a DGL graph
            Graph            = dgl.from_scipy(tempGraph)
            feat_g           = torch.tensor(tempFeats[j]).float()
            Graph.ndata['h'] = feat_g 
            graphs.append(Graph)      
            # converts each label to a torch tensor
            labels.append(torch.tensor(label[i][j], dtype = torch.long))

        # updates the progress bar
        bar.next()
    bar.finish()    
    return (graphs, label)




def perform_collation(samples):
    # The input 'samples' is a list of pairs (graph, label).
    graphs, labels, domains = map(list, zip(*samples))
    batched_graph           = dgl.batch(graphs)
    batched_graphg          = dgl.add_self_loop(batched_graph)
    return batched_graph, torch.tensor(labels), torch.tensor(domains)



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

@staticmethod
def sigmoid(z):
    return 1/(1+np.exp(-z)) 


class FocalLoss(nn.Module):
    """Computes the focal loss between input and target

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0])
        weights = target * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int
            ) -> Tensor:
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes)
        #target = self.target.to(x.device)
        weights = self._get_weights(target).to(x.device)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask)

    def _reduce(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
