import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import cv2
import numpy as np
from tools import addModelName, getXML, IoUcalculator
import ExifTags
import json


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            open_cv_image = np.array(r_image)
            # Convert RGB to BGRdata/test
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            open_cv_image = cv2.resize(open_cv_image,(800,600))
            cv2.imshow("image", open_cv_image)
            cv2.waitKey()
            break

    yolo.close_session()

FLAGS = None

def check_orientation(IMG):
    """ konwertuje obraz z formatu PIL na OpenDV"""
    exif = dict((ExifTags.TAGS[k], v) for k, v in IMG._getexif().items() if k in ExifTags.TAGS)
    return exif["Orientation"]

def PILtoOCV(IMG):
    """ Function converts PIL format to openCV"""
    open_cv_image = np.array(IMG)
    return open_cv_image[:, :, ::-1].copy()

def drawRectangle(IM, tl, br, type):
    """ Funckja rysuje prostokąt i etyiety prediction GT """

    if type:
        colour = (0, 255, 0)
        pt = br
        text = "GT"
    else:
        colour = (0, 0, 255)
        pt = tl
        text = "PREDICTION"

    IM = cv2.rectangle(IM, tl, br, colour, 7)
    FONT = cv2.FONT_HERSHEY_COMPLEX
    IM = cv2.putText(IM, text,  pt, FONT, 3, colour, 4)

    return IM


def check_sample(sample, path, yol):
    """ sprawdza próbke składającą się z pary sample[0] - IMG sample[1] - XML"""

    image = Image.open(path + "IMG/" + sample[0])
    OCV_Image = PILtoOCV(image)

    GT_Boxes = getXML(path + "XML/", sample[1])

    orientation = check_orientation(image)

    if orientation in [1, 6, 8]:   # ustawienie pionowe i pionowe obrócone o 180
        r_image, result = yol.detect_image(image)
        boxes, scores, classes = result
        ious = []

        h, w, d = OCV_Image.shape

        for i, PRbox in enumerate(boxes):
            Prediction = [int(PRbox[1]), int(PRbox[0]), int(PRbox[3]), int(PRbox[2])]  # Preciction(minx, miny, manx, maxy)

            for GTbox in GT_Boxes:
                GTCorrected = []

                if orientation == 6 :  # PIL oryginalnie odczytuje IMG obrócony o 90 st
                    GTCorrected = [int(GTbox[2]), h - int(GTbox[1]), int(GTbox[3]), h - int(GTbox[0])]
                elif orientation == 8:
                    GTCorrected = [w - int(GTbox[3]), int(GTbox[0]), w - int(GTbox[2]), int(GTbox[1])]
                else:
                    GTCorrected = [int(GTbox[0]), int(GTbox[2]), int(GTbox[1]), int(GTbox[3])]

                IoU = IoUcalculator(GTCorrected, Prediction)
                ious.append(IoU)

                OCV_Image = drawRectangle(OCV_Image,
                                          (GTCorrected[0],GTCorrected[1]),
                                          (GTCorrected[2],GTCorrected[3]),
                                          True)

                OCV_Image = drawRectangle(OCV_Image,
                                          (Prediction[0], Prediction[1]),
                                          (Prediction[2], Prediction[3]),
                                          False)

        results = [boxes, scores, classes, ious]
        return OCV_Image, results

    else:
        print("Nieznana orientacja")
        return -1


def saveIMG(IM, analys, file_name):
    """ Tworzy plik graficzny w katalogu results/nazwa_analizy/"""
    IM = cv2.resize(IM, (800, 600))
    cv2.imwrite("Results/" + analys + "/" + file_name, IM)


def createRecord(res, f):
    """ Tworzy rekord w postaci listy słowników"""
    boxes, scores, classes, ious = res
    Record = []
    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            partialRekord = dict(src=os.path.abspath(f),
                          name=os.path.basename(f),
                          score=float(scores[i]),
                          iou=float(ious[i]))
            Record.append(partialRekord)

    else:
        partialRekord = dict(src=os.path.abspath(f),
                      name=os.path.basename(f),
                      score="None",
                      iou="None")
        Record.append(partialRekord)

    return Record


def run_serial(yolo):
    """ Uruchamianie analizy dla grupy plików IMG+XML"""

    DataPath= "E:/PycharmProjects/Yolo_Kuba/Dataset/GOTOWY/D_A_I/testowy/"

    #DataPath = "E:/PycharmProjects/Yolo_Kuba/Dataset/GOTOWY/D_B_I/testowy/"
    #DataPath = "E:/PycharmProjects/keras-yolo3-master/sam/"

    A_file, A_name = addModelName(yolo.__dict__["model_path"])

    IM_files = [fl for fl in os.listdir(DataPath + "IMG") if os.path.isfile(os.path.join(DataPath + "IMG", fl))]
    AN_files = [fl for fl in os.listdir(DataPath + "XML") if os.path.isfile(os.path.join(DataPath + "XML", fl))]

    samples = zip(IM_files, AN_files)
    for cnt, val in enumerate(samples):
        print("Processing file :" + str(cnt + 1) + " z " + str(len(IM_files)))

        try:
            IM, results = check_sample(val, DataPath, yolo)

        except:
            print("błędna orientacja w plikach")
            break

        else:
            saveIMG(IM, A_name, val[0])

            rec = createRecord(results, val[0])

            A_file.write(json.dumps(rec))
            A_file.write(',\n')

    A_file.write('{}]')
    A_file.close()


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default="n",
        help = "Video input path"
    )

    parser.add_argument(
        "--serial", type=str, default="n",
        help="Test dataset path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """

        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))

    if FLAGS.__dict__["input"] != "n":
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)

    if FLAGS.__dict__["serial"] != "n":
        run_serial(YOLO(**vars(FLAGS)))

    else:
        print("Must specify at least video_input_path.  See usage with --help.")
