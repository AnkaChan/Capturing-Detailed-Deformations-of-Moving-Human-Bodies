import json, glob, os
from CNNQuadDetector import *
from os.path import join
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import tqdm

def preAccept(avgIntensity, stdDev, numDark, cfg = {'numDarkRange':(0, 1600), 'stdDevRange':(0, 60), 'intensityRange':(35, 170)}):
    if numDark < cfg['numDarkRange'][0] or numDark > cfg['numDarkRange'][1]:
        return False
    if stdDev < cfg['stdDevRange'][0] or stdDev > cfg['stdDevRange'][1]:
        return False
    if avgIntensity < cfg['intensityRange'][0] or avgIntensity > cfg['intensityRange'][1]:
        return False
    return True


def visualizeData(imgs, acceptedIndices, file):
    gridH = 10
    gridW = 10
    fig, axs = plt.subplots(gridH, gridW)

    fig.set_size_inches(20, 20 * (gridH / gridW))
    for imgId in range(gridW * gridH):
        iR = int(imgId / gridW)
        iC = int(imgId % gridH)
        axs[iR, iC].imshow(np.squeeze(imgs[imgId, :, :, :]), cmap="gray")
        titleName = "Accepted" if imgId in acceptedIndices else "Rejected"
        axs[iR, iC].set_title(titleName)
        axs[iR, iC].axis('off')

    # fig.show()
    fig.savefig(file, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)

    plt.close(fig)


if __name__ == "__main__":
    # convertedFolder = r'Z:\2019_12_13_Lada_Capture\Converted'
    convertedFolder = r'F:\WorkingCopy2\2019_12_24_SpeedTest\Converted'

    outputFolder = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\QuadStatistics'
    visFolder = join(outputFolder, 'Vis')

    os.makedirs(visFolder, exist_ok=True)
    os.makedirs(outputFolder, exist_ok=True)

    numSamples = 500
    # prepareData = False
    prepareData = True

    darkThreshold = 35

    folders = glob.glob(join(convertedFolder, '*'))
    print(folders)


    if prepareData:
        files = []

        for folder in folders:
            files.extend(glob.glob(join(folder, '*.pgm')))
        # files = json.load(open('AllFiles.json'))

        print("Number of total files: ", len(files))

        # np.random.shuffle(files)

        qDetector = CNNQuadDetector()
        qDetector.restoreCNNSess()
        qDetector.cfg = CNNQuadDetectorCfg()

        for f in tqdm.tqdm(files[:numSamples]):
            imgFName = Path(f).stem

            print("Processing: ", f)
            # path = str(imgFilePath.parent)

            start = time.clock()
            qDetector.imgFile = f
            qDetector.img = cv2.imread(f)

            print('Time consumption in reading image', time.clock() - start)
            qDetector.process2()

            # outQuadImgs = np.array(qDetector.img_list)
            # outQuadImgsFile = join(outputFolder, imgFName + '.npy')
            # np.save(outQuadImgsFile, outQuadImgs)

            outInfoFile = join(outputFolder, imgFName + '.json')

            # json.dump({
            #     'qv_list':[vs.tolist() for vs in  qDetector.qv_list],
            #     'accept_indices':qDetector.accept_indices.tolist(),
            #     'accept_qi': qDetector.accept_qi.tolist(),
            #     'accept_qv': qDetector.accept_qv.tolist(),
            #     'recog_dens4a': qDetector.recog_dens4a.tolist(),
            #     'recog_dens4b': qDetector.recog_dens4b.tolist()
            # }, open(outInfoFile, 'w'), indent=2)

    quadFiles = glob.glob(join(outputFolder, '*.npy'))

    avgIntensityAccepted = []
    avgIntensityRejected = []

    stdDevAccepted = []
    stdDevRejected = []

    numDarkAccepted = []
    numDarkRejected = []

    numPreAccepted = 0
    numPreRejected = 0


    for qf in tqdm.tqdm(quadFiles[:]):
        imgFName = Path(qf).stem
        try:
            imgs = np.load(qf)
            data = json.load(open(join(outputFolder, imgFName + '.json')))

        except:
            continue

        outputVisFile = join(visFolder, imgFName + '.pdf')
        # visualizeData(imgs, data, outputVisFile)

        acceptedIds = data['accept_indices']
        rejectedIds = np.setdiff1d(range(imgs.shape[0]), acceptedIds) #[i if i in data['accept_indices'] for i in range(imgs.shape[0])]

        for i in acceptedIds:
            avgIntensityAccepted.append(np.mean(imgs[i, 20:84, 20:84]))
            stdDevAccepted.append(np.std(imgs[i, 20:84, 20:84]))
            numDarkAccepted.append(np.count_nonzero(imgs[i, 20:84, 20:84] < darkThreshold))

            if preAccept(avgIntensityAccepted[-1], stdDevAccepted[-1], numDarkAccepted[-1]):
                numPreAccepted = numPreAccepted + 1
            else:
                numPreRejected = numPreRejected + 1

        for i in rejectedIds:
            avgIntensityRejected.append(np.mean(imgs[i, 20:84, 20:84]))
            stdDevRejected.append(np.std(imgs[i, 20:84, 20:84]))
            numDarkRejected.append(np.count_nonzero(imgs[i, 20:84, 20:84] < darkThreshold))

            if preAccept(avgIntensityRejected[-1], stdDevRejected[-1], numDarkRejected[-1]):
                numPreAccepted = numPreAccepted + 1
            else:
                numPreRejected = numPreRejected + 1


    print("numPreAccepted:", numPreAccepted)
    print("numPreRejected", numPreRejected)
    print("Percentage Prerejected:", 100*numPreRejected / (numPreAccepted + numPreRejected))

    # fftOld = np.abs(rfft(xTrajectoryOld))
    # fftRefined = np.abs(rfft(xTrajectoryRefined))
    # numFrames = fftOld.shape[0]
    # fig, ax = plt.subplots()
    # x = np.arange(fftOld.shape[0])  # the label locations
    # width = 0.5
    # rects1 = ax.bar(x - width / 2, (fftRefined), width, label='fftRefined', log=True)
    # rects2 = ax.bar(x + width / 2, (fftOld), width, label='fftOld', log=True)
    # ax.legend()
    #
    bins = np.linspace(0, 255, 100)

    plt.hist(avgIntensityAccepted, bins=bins, alpha=0.5, label='avgIntensityAccepted', log=True)
    plt.hist(avgIntensityRejected, bins=bins, alpha=0.5, label='avgIntensityRejected', log=True)
    plt.legend(loc='upper right')

    plt.figure()
    plt.hist(stdDevAccepted, bins=bins, alpha=0.5, label='stdDevAccepted', log=True)
    plt.hist(stdDevRejected, bins=bins, alpha=0.5, label='stdDevRejected', log=True)
    plt.legend(loc='upper right')

    binsNumDark = np.linspace(0, 10000, 100)

    plt.figure()
    plt.hist(numDarkAccepted, bins=binsNumDark, alpha=0.5, label='numDarkAccepted', log=True)
    plt.hist(numDarkRejected, bins=binsNumDark, alpha=0.5, label='numDarkRejected', log=True)
    plt.legend(loc='upper right')


    plt.show()
