import os, datetime
import xml.etree.ElementTree as ET


def createAnalysysFile(model):
    """ Funkcja tworzy plik txt przechowujęcy rekordy"""

    now = datetime.datetime.now()
    name = model + now.strftime("-%Y-%m-%d_%H-%M")
    if not os.path.isdir("Results/"):
        print("Creating missing path: Results/" + name)
        os.mkdir("Results")

    if not os.path.isdir("Results/" + name):
        print("Creating missing path: Results/" + name)
        os.mkdir("Results/" + name)

    file = open("Results/" + name + "/" + name + ".txt", "w")
    file.write('[')
    return file, name

def XML_reader(tree):
    """ Odczytywanie parametru bndbox z pliku XML"""
    root = tree.getroot()
    all_boxes = []

    for box in root.iter('bndbox'):
        boxPt = [box[0].text,box[1].text,box[2].text,box[3].text]
        all_boxes.append(boxPt)

    return all_boxes


def getXML(path, file):
    """ Wczytanie pliku XML """

    print("Odczyt pliku "+path+file)
    tree = ET.parse(path+file)
    return XML_reader(tree)


def IoUcalculator(GT_Box, PR_Box):
    """ Obliczanie wartości IOU dla dwoch obszarow"""
    """ boxy podawane w formacie (minx, miny, manx, maxy)"""

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(GT_Box[0], PR_Box[0])
    yA = max(GT_Box[1], PR_Box[1])
    xB = min(GT_Box[2], PR_Box[2])
    yB = min(GT_Box[3], PR_Box[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (GT_Box[2] - GT_Box[0] + 1) * (GT_Box[3] - GT_Box[1] + 1)
    boxBArea = (PR_Box[2] - PR_Box[0] + 1) * (PR_Box[3] - PR_Box[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def addModelName(model_path):
    _, name = model_path.split("/")
    name, ext = name.split(".")

    return createAnalysysFile(name)