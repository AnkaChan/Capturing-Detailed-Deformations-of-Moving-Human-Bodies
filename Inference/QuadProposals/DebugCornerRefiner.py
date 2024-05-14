from CNNQuadDetector import *
import cv2
from os.path import join
from pathlib import Path

if __name__ == "__main__":
    inDebugImgFiles = [
        r'Z:\2019_08_09_AllPoseCapture\Converted\A\15782.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\A\15794.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\A\16023.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\A\16598.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\A\16647.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\A\16790.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\A\16796.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\B\15602.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\B\15709.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\B\15715.pgm',
        r'Z:\2019_08_09_AllPoseCapture\Converted\B\15840.pgm'
    ]

    tf.reset_default_graph()
    cornerdet_sess = r'C:\Code\MyRepo\chbcapture\08_CNNs\QuadProposals\nets\28_renamed.ckpt'
    outputDebugFolder = r'Z:\2019_08_09_AllPoseCapture\DebugCornerRefiner'
    saver = tf.train.import_meta_graph(cornerdet_sess + '.meta', import_scope="cornerdet")
    sess_cornerdet = tf.Session()
    saver.restore(sess_cornerdet, cornerdet_sess)

    for imgF in inDebugImgFiles:
        img = cv2.imread(imgF)

        crops, i_list, j_list = qps.gen_crops(img)

        (d4a, d4b) = ia.cornerdet_inference(sess_cornerdet, crops)
        assert (d4a.shape[0] == d4b.shape[0] and len(i_list) == d4a.shape[0] and len(j_list) == d4a.shape[0])

        corners1, confids = ia.extract_corners_confids(d4a, d4b, i_list, j_list)
        corners1, confids = qps.cluster_points_with_confidences(corners1, confids)

        corners2 = ia.refine_corners(sess_cornerdet, corners1, img)

        diff = corners2 - corners1
        dis = np.sqrt(np.square(diff[:,0]) + np.square(diff[:,1]))
        print("Max dis: ", np.max(dis), "Max dis corner id", np.argmax(dis))
        cid = np.argmax(dis)
        outPdfFile = join(outputDebugFolder, Path(imgF).stem + '.pdf')

        fig, ax = plt.subplots()
        ax.imshow(img,  cmap="gray")
        ax.plot(corners1[cid, 0], corners1[cid, 1], 'x', color='red', markeredgewidth = 0.06, markersize=0.3)
        ax.plot(corners2[cid, 0], corners2[cid, 1], 'x', color='blue', markeredgewidth = 0.06, markersize=0.3)

        fig.savefig(outPdfFile, dpi = 2000,  transparent = True, bbox_inches = 'tight', pad_inches = 0)