# -*- coding: utf-8 -*-
"""
"""
def model_train(model, optimizer, scheduler, trainset, testset, nepochs, batch_size, Adversarial, d_train, d_test):
    """
    Train the model with the given parameters and datasets.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        scheduler: The learning rate scheduler.
        trainset: The training dataset.
        testset: The test dataset.
        nepochs: The number of training epochs.
        batch_size: The batch size for training.
        Adversarial: Flag indicating whether to use adversarial training.
        d_train: The training domain labels.
        d_test: The test domain labels.

    Returns:
        model: The trained model.
        test_Y: The test labels.
        score_Y_best: The best score on the test set.
        att_test: The test attention scores.
    """

    from focal_loss import FocalLoss  

    device = "cuda:0"
    gradient_accumulations = 4

    # Prepare DataLoader for training data
    data_loader = DataLoader(trainset, batch_size, collate_fn=perform_collation, shuffle=True, pin_memory=True)

    # Initialize loss functions
    lossfunc_class = FocalLoss(gamma=2)
    lossfunc_domain = nn.BCEWithLogitsLoss()

    att_test = []

    # List to save epoch losses
    epochLosses = []

    # Loop over epochs
    for epoch in range(nepochs):
        epoch_loss = 0
        model.train()

        for iter, (bg, label, domain) in enumerate(data_loader):

            optimizer.zero_grad()

            p = float(iter + epoch*len(data_loader)) / 200 / len(data_loader)
            alpha = 2. / (1.+np.exp(-10*p)) - 1

            bg = bg.to(device)
            feat = bg.ndata['h']

            class_output, domain_output, att = model(bg, alpha, device)

            label = label.type_as(class_output)
            class_output = torch.reshape(class_output, label.size())
            loss_class = lossfunc_class(class_output, label.long())

            if Adversarial:
                domain = torch.reshape(domain, domain_output.size())
                domain = label.type_as(domain_output)
                loss_domain_s = lossfunc_domain(domain_output, domain)

                test_bg, td_output, att = model(test_bg, alpha, device, get_attention=True)
                d_test = d_test.to(device)
                loss_domain_t = lossfunc_domain(td_output, d_test)

                loss = loss_class + loss_domain_s + loss_domain_t
            else:
                loss = loss_class 

            loss.backward()
            optimizer.step()
            model.zero_grad()

            epoch_loss += loss.detach().item()

        model.eval()

        with torch.no_grad():
            test_X, test_Y, test_d = map(list, zip(*testset))
            test_bg = dgl.batch(test_X).to(device)
            test_Y = torch.tensor(test_Y, device='cpu').float().view(-1, 1)

            score_Y, domain_output, att_test = model(test_bg, alpha, device)
            score_Y = torch.tensor(torch.sigmoid(score_Y), requires_grad=False, device='cpu')

            AUC_test = round(roc_auc_score(test_Y, score_Y), 3)

            if epoch == 0:
                AUC_test_best = AUC_test
                score_Y_best = score_Y
            elif AUC_test > AUC_test_best:
                AUC_test_best = AUC_test
                score_Y_best = score_Y

            epoch_loss /= (iter + 1)
            print('Epoch {}, loss {:.4f}, AUC_test {:.3f}'.format(epoch, epoch_loss, AUC_test))

            epochLosses.append(epoch_loss)

    del test_X, bg, test_bg 
    return model, test_Y, score_Y_best, att_test



def ablation(trainset, testset, model, layer, classifier, conv_chans, conv_width):
    """
    Perform ablation study by selectively disabling some parts of the model and assessing the impact on performance.

    Args:
        trainset: The training dataset.
        testset: The test dataset.
        model: The model to be ablated.
        layer: The layer of the model to perform ablation on.
        classifier: The classifier used in the model.
        conv_chans: Number of convolutional channels.
        conv_width: Width of the convolutional layer.

    Returns:
        AUC_abl_ts: AUC score of the ablated model on testset.
        kappa: Kappa score of the ablated model on testset.
    """
    # Function to register hooks
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register forward hook
    model.conv8.register_forward_hook(get_activation(layer))

    # Initialize the ablated model, optimizer, scheduler and loss function
    model_ablation = classifier_ablation(conv_chans, conv_width).to("cuda:0")
    optimizer = optim.Adam(model_ablation.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_func = nn.BCEWithLogitsLoss()

    # Prepare DataLoader for training data
    data_loader_abl = DataLoader(trainset, batch_size=256, collate_fn=perform_collation, shuffle=False, pin_memory=True)
    device = "cuda:0"

    # Training loop
    for epoch in range(40):
        epoch_loss = 0

        for i, (X_train, y, domain) in enumerate(data_loader_abl):
            model_ablation.train()
            optimizer.zero_grad()

            X_train = X_train.to(device)
            model.to(device)
            score, domain_output, att = model(X_train, alpha, device=device)
            act = activation[layer]

            model_ablation = model_ablation.to(device)
            score_output = model_ablation(act, X_train)

            # Reshape and type cast y to match score_output
            y = torch.reshape(y, score_output.size())
            y = y.type_as(score_output)

            # Compute loss and backpropagate
            loss = loss_func(torch.nn.Sigmoid(torch.nan_to_num(score_output, nan=0.25)), y.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()

        # Switch to evaluation mode
        model_ablation.eval()

        with torch.no_grad():
            test_X, test_Y, test_d = map(list, zip(*testset))
            test_bg = dgl.batch(test_X).to("cpu")

            model = model.to("cpu")
            model_ablation = model_ablation.to("cpu")
            score_ts, domain_output_ts, att = model(test_bg, alpha, device="cpu")

            act_ts = activation[layer].to("cpu")
            score_abl_ts = model_ablation(act_ts, test_bg)

            score_abl_ts = torch.tensor(torch.sigmoid(score_abl_ts), requires_grad=False, device='cpu')

            # Compute the performance metrics
            AUC_abl_ts = round(roc_auc_score(test_Y, score_abl_ts), 3)
            post_scores, acc, sen, spec, tpr, fpr, auc, auc_sk, auc_sk90, gdrs, fdhs, fdurs, kappa = compute_performance(
                (np.array(score_abl_ts)).squeeze(), (np.array(test_Y)).squeeze(), 0, 20)

            model_ablation = model_ablation.to(device)

        epoch_loss /= (i + 1)

    return AUC_abl_ts, kappa
