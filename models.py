# -*- coding: utf-8 -*-


class Classifier3(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, domain_adapt):
        super(Classifier3, self).__init__()
        self.domain_adapt = domain_adapt
        self.conv1       = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size= [1,5], stride= 1) 
        self.conv2       = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size= [1,5], stride= 1) # out: 188 to 184 , 8 channels
        self.pool1       = torch.nn.MaxPool2d(kernel_size =[1,2], stride=[1,2]) # out: 92
        self.bn1         = torch.nn.BatchNorm2d(8)
        self.conv3       = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size= [1,5], stride= 1) # out: 92 to 88
        self.conv4       = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size= [1,5], stride= 1) # out168 to 172
        self.pool2       = torch.nn.MaxPool2d(kernel_size =[1,2], stride=[1,2]) # out: 180 to 86
        self.bn2         = torch.nn.BatchNorm2d(32)
        self.conv5       = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size= [1,5], stride= 1) #86
        self.conv6       = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size= [1,5], stride= 1) #82
        self.pool3       = torch.nn.MaxPool2d(kernel_size =[1,2], stride=[1,2]) # out: 41
        self.bn3         = torch.nn.BatchNorm2d(8)
        self.conv7       = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size= [1,5], stride= 1) # 37
        self.conv8       = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size= [1,5], stride= 1) # 34
        self.pool4       = torch.nn.MaxPool2d(kernel_size =[1,2], stride=[1,2]) # out: 41
        
        self.gconv1      =dgl.nn.pytorch.conv.GATConv(33, 16, num_heads=1, feat_drop=0.2)
        self.gconv2      =dgl.nn.pytorch.conv.GATConv(16, 16, num_heads=1, feat_drop=0.2)
        self.gconv3      =dgl.nn.pytorch.conv.GATConv(16, 16, num_heads=1, feat_drop=0.2)
  
        self.gate_nn   = nn.Linear(16, 1)
        self.att       = dgl.nn.pytorch.glob.GlobalAttentionPooling(self.gate_nn)
        self.maxpool    = dgl.nn.pytorch.glob.MaxPooling()
        self.lin1      = nn.Linear(16,16)   
        self.lin2      = nn.Linear(16, 4)   
        self.classify  = nn.Linear(4, 1) 
        self.sigm      = torch.nn.Sigmoid()
        self.domain_classifier = nn.Linear(16,1)


    def forward(self, g, alpha, device, get_attention=True):

      h   = g.ndata["h"].to(device)
      g.to(device)
      batchsize = h.size(0)
      h = h.unsqueeze(0).unsqueeze(0)
      t1 = time.time()
      #print(h.shape)
      h = self.conv1(h)
      h = self.conv2(h)
      h = self.pool1(h)
      h = F.relu(self.bn1(h))
      h = self.conv3(h)
      h = self.conv4(h)
      h = self.pool2(h)
      h = F.relu(self.bn2(h))
      h = F.relu(self.conv5(h))
      h = F.relu(self.conv6(h))     
      h = self.pool3(h)
      h = F.relu(self.bn3(h))
      h = self.conv7(h)
      h = F.relu(self.conv8(h))
      h = h.squeeze(0).squeeze(0)
      t2 = time.time()

      h    ,att  = self.gconv1(g, h, get_attention=True)
      h     = F.relu(h)
      h    = self.gconv2(g, h)
      h     = F.relu(h)
      h      = self.gconv3(g, h)

      g.ndata['h'] = h
      hg = dgl.mean_nodes(g, 'h')

      domain_output = None
      if self.domain_adapt:
          reverse_h    = ReverseLayerF.apply(hg, alpha)
          domain_output= self.domain_classifier(reverse_h)

      hg  = F.relu(self.lin1(hg))
      hg  = self.lin2(hg)
      class_output = self.classify(hg)
      class_output = self.sigm(class_output)

      return torch.reshape(class_output,(class_output.size(0),1)), domain_output,att #
  

    
class classifier_ablation(nn.Module):
    def __init__(self, conv_chans, conv_width):
        super(classifier_ablation, self).__init__()
        self.conv1       = torch.nn.Conv2d(in_channels=conv_chans, out_channels=1, kernel_size= [1,1], stride= 1)
        self.conv2       = torch.nn.AvgPool2d(kernel_size= [18,1], stride= [18,1])
        self.lin1        = nn.Linear(conv_width,16)
        self.lin2        = nn.Linear(16,4)
        self.Drop        = nn.Dropout(p=0.2)
        self.classify    = nn.Linear(4,1)

    def forward(self, h,g):
        h   = self.conv1(h)
        h   = self.conv2(h)
        h  = self.lin1(h.squeeze(0).squeeze(0))
        h  = F.leaky_relu(self.lin2(h))
        #h  = self.Drop(h)
        return(self.classify(h))
     
        
     
class classifier_ablation_gnn(nn.Module):
    def __init__(self,conv_chans, conv_width):
        super(classifier_ablation_gnn, self).__init__()
        self.lin1        = nn.Linear(conv_width,16)
        self.lin2        = nn.Linear(16,4)
        self.Drop        = nn.Dropout(p=0.2)
        self.classify    = nn.Linear(4,1)

    def forward(self, h,g):
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        hg  = self.lin1(hg)
        hg  = F.leaky_relu(self.lin2(hg), negative_slope=0.3)
        return self.classify(hg)