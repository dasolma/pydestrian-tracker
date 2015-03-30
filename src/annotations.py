__author__ = 'dasolma'

import os
import re
from os import listdir
from os.path import isfile, join
from os.path import basename

class PascalAnnotations():


    def __init__(self, path):
        self.path = path
        self.objects = {}
        self.current_file_objects = {}

        files = [ f for f in listdir(path) if isfile(join(path,f)) ]

        for file in files:
            with open(join(path,file)) as fp:
                for line in iter(fp.readline, ''):
                    self.process_line(line)

            self.objects[self.current_file] = self.current_file_objects



    def process_line(self, line):
        if "Image filename :" in line:
            file = line.split(':')[1].replace("\"", "").strip()
            self.current_file = basename(file)
            self.current_file_objects = {}
            return

        if "# Details for object 1 (" in line:
            object = line.split('(')[1].replace("\"", "").replace(")", "").strip()
            self.current_obj = object

            return

        if "Bounding box for object" in line:
            bb  = re.split(",|-", line.split(':')[1].replace(")", "").replace("(", ""))
            (x1,y1, x2, y2) = [x.strip() for x in bb]

            if not self.current_obj in self.current_file_objects.keys():
                self.current_file_objects[self.current_obj] = []

            self.current_file_objects[self.current_obj].append( (x1,y1, x2, y2) )






#pa = annotations.PascalAnnotations('../data/raw/INRIAPerson/Test/annotations')
