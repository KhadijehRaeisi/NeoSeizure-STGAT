# -*- coding: utf-8 -*-
"""

"""


# define functions
def compute_performance(scores, targets, collar, movingaverage):
    # This function computes the performance measure

    # INPUTS:
    #   scores        : is a torch tensor containing the raw values of the classifier's outpu in 1 Hz
    #   targets       : is a torch tensor containing the lables provided by experts in 1 Hz
    #   collar        : is the length of collaring in the post-processing in second
    #   movingaverage : is the length of moving average filter in second

    # OUTPUTS:
    #   acc  : is a numpy array containing 1000 values of accuracy corresponding to 1000 different threshold values
    #   sen  : is a numpy array containing 1000 values of sensitivity corresponding to 1000 different threshold values
    #   spec : is a numpy array containing 1000 values of specificity corresponding to 1000 different threshold values
    #   tpr  : is a numpy array containing 1000 values of true positive rate corresponding to 1000 different threshold values
    #   fpr  : is a numpy array containing 1000 values of false positive rate corresponding to 1000 different threshold values
    #   gdr  : is the percentage of seizure events correctly identified by the system as labelled by an expert in neonatal EEG. If a
    #           seizure was detected any time between the start and end of a labelled seizure this was considered a good detection.
    #   fdh  : number of false detections per hour (FD/h) calculated as the number of predicted seizure events in 1 h that
    #          have no overlap with actual reference seizures.       

    # converts torch tensor to numpy array and makes them as 1D numpy arrays
    
    #targets = np.reshape(targets.numpy(), targets.numpy().size)

    # post-processing
    # moveing average filter with size of "movingaverage" seconds 
    if movingaverage == 0:
        dummyScores = scores   
    else:
        dummyScores = np.convolve(scores, np.ones(movingaverage), 'same') / movingaverage
        

    # gets max and min of values of scores
    minScore = 0 #dummyScores.min() 
    maxScore = 1 #dummyScores.max()

    # creates a range of thresholds for evaluation of the alforithm
    thresholds = np.linspace(minScore, maxScore, 50) #####################################################change 10 to 1000
    acc = np.zeros([])
    sen = spec = fpr = tpr = gdr = fdh = fdur = kappa = acc

    # computes performance measures over the threshold range
    for i in range(thresholds.size):
        binarizedScores = np.zeros(dummyScores.size)
        binarizedScores[dummyScores > thresholds[i]] = 1

        # performs post-processing
        processedBinarizedScores = perform_postprocessing(binarizedScores, collar)

        # calculates the performance metrics
        ACC ,SEN , SPEC = compute_acc_sen_spec(targets, processedBinarizedScores)
        GDR , FDH, FDur = compute_gdr_fdh_fdur(targets, processedBinarizedScores)
        KAPPA    = round(cohen_kappa_score(targets, processedBinarizedScores,labels=None), 3)

        acc  = np.r_[acc, ACC]
        sen  = np.r_[sen, SEN]  
        spec = np.r_[spec, SPEC]
        tpr  = np.r_[SEN, tpr]
        fpr  = np.r_[(1 - SPEC), fpr]
        gdr  = np.r_[gdr, GDR]
        fdh  = np.r_[fdh, FDH]
        fdur = np.r_[fdur, FDur]
        kappa= np.r_[kappa, KAPPA]
  
    # to ensure (0,0) is included
    tpr  = np.r_[0, tpr]
    fpr  = np.r_[0, fpr]
    fpr  = np.round(fpr,  4)
    tpr  = np.round(tpr,  4)

    # to ensure (1,0) is included
    dummyFPR = np.append(fpr[:-1], 1)
    dummyTPR = np.append(tpr[:-1], 0)
    auc = compute_auc(dummyFPR, dummyTPR)

    auc_sk = round(roc_auc_score(targets, dummyScores), 3)
    # max_fprfloat > 0 and <= 1, default=None,
    #If not None, the standardized partial AUC over the range [0, max_fpr] is returned
    auc_sk90 = round(roc_auc_score(targets, dummyScores, max_fpr=0.1), 3)
    

    return (processedBinarizedScores, acc, sen, spec, tpr, fpr, auc, auc_sk, auc_sk90, gdr, fdh, fdur, kappa)



def perform_postprocessing(binarizedScores, collar):
    # This function performs two postprocessing steps: elimination of segemnts whose lengths are less than 10 
    # seconds, and performs collaring
    
    # This function eliminates seizure segments whose length is less than "lim" seconnds

    # INPUTS:
    #   binarizedScores  : is a numpy array
    #   collar           : is the added length to each siezure segment

    # OUTPUT:
    #   processedBinarizedScores : is the modified version of the input array which does not consist seizures whose length is less than lim second

    # eliminates short seizure segments
    lim = 10
    dummyBinarizedScores         = np.zeros(binarizedScores.size)
    zeroAddedBinarizedScores     = np.append(np.insert(binarizedScores, 0, 0), 0)
    diffZeroAddedBinarizedScores = np.diff(zeroAddedBinarizedScores,1)
    beginningPoints1             = np.where(diffZeroAddedBinarizedScores == 1)[0]
    endingPoints1                = np.where(diffZeroAddedBinarizedScores == -1)[0] 
    lessThan10                   = np.where((endingPoints1 - beginningPoints1) < lim)[0]
    beginningPoints1[lessThan10] = 0
    endingPoints1[lessThan10]    = 0
    
    for i in range(beginningPoints1.size):
        dummyBinarizedScores[beginningPoints1[i] : endingPoints1[i]] = 1

    # performs collaring
    processedBinarizedScores          = np.zeros(dummyBinarizedScores.size)
    zeroAddedDummyBinarizedScores     = np.append(np.insert(dummyBinarizedScores, 0, 0), 0)
    diffZeroAddedDummyBinarizedScores = np.diff(zeroAddedDummyBinarizedScores,1)
    beginningPoints2                  = np.where(diffZeroAddedDummyBinarizedScores == 1)[0]
    endingPoints2                     = np.where(diffZeroAddedDummyBinarizedScores == -1)[0] - 1
    endingPoints2                     = endingPoints2 + collar
    endingPoints2[endingPoints2 > processedBinarizedScores.size] = processedBinarizedScores.size
    for i in range(endingPoints2.size):
        processedBinarizedScores[beginningPoints2[i] : endingPoints2[i] + 1] = 1
    return processedBinarizedScores



def compute_acc_sen_spec(targets, scores):
    # This function computes accuracy, sensitivity, and specificty
    
    # INPUTS:
    #   targets : a tensor containing annotations provided by experts (composed of 0s and 1s)
    #   scores  : a tensor containing the output of classifier (composed of 0s and 1s)

    # OUTPUT:
    #   acc  : accuracy of the classifier [0,1]
    #   sen  : sensitivity of the classifier [0,1]
    #   spec : specificity of the classifier [0,1]
    TP = FP = TN = FN = 0    
    for i in range(len(scores)): 
        if targets[i] == 1 and scores[i] == targets[i]:
            TP += 1
        if scores[i] == 1 and targets[i] != scores[i]:
            FP += 1
        if targets[i] == 0 and scores[i] == targets[i]:
            TN += 1
        if scores[i] == 0 and targets[i] != scores[i]:
            FN += 1

    acc  = ((TP + TN) / (TP + FN + FP + TN)) 
    sen  = (TP / (TP + FN))
    spec = (TN / (TN + FP))
    return(acc, sen, spec)



def compute_auc(fpr, tpr):
    # This function computes the area under the ROC curve

    # INPUTS:
    #   fpr : a numpy array containg fpr values for different threshold values
    #   tpr : a numpy array containg tpr values for different threshold values

    # OUTPUT:
    #   AUC : area under the ROC curve

    AUC = 0.5*np.abs(np.dot(fpr,np.roll(tpr,1))-np.dot(tpr,np.roll(fpr,1)))
    return AUC



def compute_gdr_fdh_fdur(targets, scores):
    trueDetected  = 0
    falseDetected = 0
    falseDuration = 0

    targetsZeroAdded       = np.append(np.insert(targets, 0, 0), 0)
    targetsDiff            = np.diff(targetsZeroAdded, 1)
    targetsBeginningPoints = np.where(targetsDiff == 1)[0]
    targetsEndingPoints    = np.where(targetsDiff == -1)[0] - 1
    targetsEventLimits     = np.vstack((targetsBeginningPoints, targetsEndingPoints))
    targetsEventsNumber    = targetsEventLimits.shape[1]

    scoresZeroAdded       = np.append(np.insert(scores, 0, 0), 0)
    scoresDiff            = np.diff(scoresZeroAdded, 1)
    scoresBeginningPoints = np.where(scoresDiff == 1)[0]
    #print(scoresBeginningPoints)
    scoresEndingPoints    = np.where(scoresDiff == -1)[0] - 1
    #print(scoresEndingPoints)
    scoresEventLimits     = np.vstack((scoresBeginningPoints, scoresEndingPoints))
    scoresEventsNumber    = scoresEventLimits.shape[1]

    if targetsEventsNumber == 0:
        trueDetected  = 0
        falseDetected = scoresEventsNumber
    else:
        if scoresEventsNumber == 0:
            trueDetected  = 0
            falseDetected = 0
        else:
            for i in range(scoresEventsNumber):
                dummyTargetEvent1 = targets[scoresEventLimits[0,i] : scoresEventLimits[1,i] + 1]
                dummyScoresEvent1 = scores[scoresEventLimits[0,i]  : scoresEventLimits[1,i] + 1]
                #dummyScoresEvent1 = torch.tensor(np.expand_dims(dummyScoresEvent1, axis=1))
                #dummyScoresEvent1 = torch.tensor(dummyScoresEvent1)
                #print(dummyScoresEvent1)
                #print(dummyTargetEvent1)
                if (dummyTargetEvent1 * dummyScoresEvent1).sum() == 0:
                    falseDetected += 1
                    falseDuration += dummyScoresEvent1.sum()

            for i in range(targetsEventsNumber):
                dummyTargetEvent2 = targets[targetsEventLimits[0,i] : targetsEventLimits[1,i] + 1]
                dummyScoresEvent2 = scores[targetsEventLimits[0,i]  : targetsEventLimits[1,i] + 1]
                if  (dummyTargetEvent2 * dummyScoresEvent2).sum() != 0:
                    trueDetected += 1
    gdr  = (trueDetected / targetsEventsNumber) * 100
    fdh  = (falseDetected / (targets.shape[0] / 3600))
    fdur = falseDuration / 60
    return gdr, fdh, fdur