import json
import itertools
from matplotlib import pyplot as plt
import numpy as np
import io
import base64

import PIL.Image

templateObjStr = '''{
  "version": "3.16.0",
  "flags": {
    "valid": true
  },
  "shapes": [
    {
      "label": "0",
      "line_color": null,
      "fill_color": null,
      "points": [
        [
          1894.5,
          496.5
        ],
        [
          1888.5,
          490.5
        ],
        [
          1900.5,
          490.5
        ],
        [
          1900.5,
          502.5
        ],
        [
          1888.5,
          502.5
        ],
        [
          1888.5,
          490.5
        ]
      ],
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "lineColor": [
    0,
    255,
    0,
    128
  ],
  "fillColor": [
    255,
    0,
    0,
    128
  ],
  "imagePath": "01732E.pgm",
  "imageData": "",
  "imageHeight": 2160,
  "imageWidth": 4000
}'''

def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get('Orientation', None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        logger.error('Failed opening image file: {}'.format(filename))
        return

    # apply orientation to image according to exif
    image_pil = apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        format = 'PNG'
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()

def _check_image_height_and_width(imageData, imageHeight, imageWidth):
    img_arr = utils.img_b64_to_arr(imageData)
    if imageHeight is not None and img_arr.shape[0] != imageHeight:
        logger.error(
            'imageHeight does not match with imageData or imagePath, '
            'so getting imageHeight from actual image.'
        )
        imageHeight = img_arr.shape[0]
    if imageWidth is not None and img_arr.shape[1] != imageWidth:
        logger.error(
            'imageWidth does not match with imageData or imagePath, '
            'so getting imageWidth from actual image.'
        )
        imageWidth = img_arr.shape[1]
    return imageHeight, imageWidth

def getLabelMeTemplate(imgFile):
    templateObj = json.loads(templateObjStr)
    imageData = load_image_file(imgFile)
    #imageHeight = imageData.shape[0]
    #imageWidth = imageData.shape[1]

    imageData = base64.b64encode(imageData).decode('utf-8')
    #templateObj['imageWidth'] = imageWidth
    #templateObj['imageHeight'] = imageHeight
    templateObj['imageData'] = imageData
    return templateObj


def loadLabelMePolygonLabels(labelFile):
    with open(labelFile, 'r') as myfile:
        #with open('wenxian_02241.json', 'r') as myfile:
        data=myfile.read()

        # parse file
        obj = json.loads(data)

        return obj['shapes'], obj


def turnToLabelMeLabelbj(labelSet, templateObj, imgFile = None):
    if imgFile != None:
        templateObj = getLabelMeTemplate(imgFile)

#     templateObj['shapes'] = []
#     for l in labelSet:
#         label =  {
#             "label": l['label'],
#             "line_color": None,
#             "fill_color": None,
#             "points": l['points'],
#             "shape_type": "polygon"
#         }
#         templateObj['shapes'].append(label)
    return templateObj

def writeAsLabelMeLabelFileWithImg(labelmeLabelFileOutFile, labelSet, imgFile):
    templateObj = turnToLabelMeLabelbj(labelSet, None, imgFile=imgFile)
    json.dump(templateObj, open(labelmeLabelFileOutFile, 'w'), indent = 2)


def writeAsLabelMeLabelFile(outFile, labelSet, templateObj):
    obj = turnToLabelMeLabelbj(labelSet, templateObj)
    json.dump(obj, open(outFile, 'w'), indent = 2)

def drawLabel(outPDFName, labelSet, img, drawText = True):
    fig, ax = plt.subplots()
    ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest', cmap=plt.get_cmap('gray'))
    for i, item in enumerate(labelSet):
        pts = np.array(item['points'])

        polygonIndices = list(range(pts.shape[0]))
        polygonIndices.append(0) 
        x_coords = pts[polygonIndices, 0]
        y_coords = pts[polygonIndices, 1]
        ax.plot(x_coords, y_coords, '-', linewidth=0.02, color='red')
        if drawText:
            ax.text(np.mean(x_coords[0:4]), np.mean(y_coords[0:4]), item['label'], \
                        verticalalignment='center', horizontalalignment='center', fontsize=1, color='green')
            ax.text(x_coords[0], y_coords[0], '0', verticalalignment='top', horizontalalignment='left', fontsize=0.3, color='red')

    fig.savefig(outPDFName, dpi = 2000)
    plt.close()
    print('Saved pdf file: ', outPDFName)