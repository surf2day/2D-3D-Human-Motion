import os
import sys
import argparse
import csv
import numpy as np
import h5py

from braniac.utils import *
from braniac.format.nturgbd.body import BodyFileReader

class DataSplitter(object):
    '''
    A helper class to split training data and generate a CSV files with the list of training data.
    '''
    def __init__(self, input_folder, exclude_file):
        '''
        Initialize DataSplitter object.

        Args:
            input_folder(str): path of the input folder that contains all the clips.
        '''
        self._input_folder = input_folder
        self._exclude_file = exclude_file
        self._items = []
        self._excludeFiles = []

        self._stats_context = DataStatisticsContext()

    def load_exclude_files(self):

        with open(self._exclude_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.rstrip("\n")
                line = line.strip()
                line = line+'.skeleton'
                self._excludeFiles.append(line)

    def load_data_paths(self):
        '''
        Load the list of files or sub-folders into a python list with their
        corresponding label.
        '''
        files = os.listdir(self._input_folder)
        index = 0
        excludeCntr = 0
        self._items.clear()
        for item in files:
            item_path = os.path.join(self._input_folder, item)
            if not os.path.isfile(item_path):
                continue
 
            if item in self._excludeFiles:
                print("excluding file_{}".format(item))
                excludeCntr += 1
                continue

            settings = self._parse_filename(item)            
            if self._filter_data(item_path, settings):
                subject_id = settings[2]
                activity_id = settings[4]
                self._items.append([os.path.abspath(item_path), activity_id, subject_id])

            if (index % 100) == 0:
                print("Process {} items.".format(index+1))
            index += 1
        
        print("{} files excluded".format(excludeCntr))
        return self._items

    def _parse_filename(self, filename):
        '''
        Each file name in the dataset is in the format of SsssCcccPpppRrrrAaaa 
        (e.g. S001C002P003R002A013), for which sss is the setup number, ccc is 
        the camera ID, ppp is the performer ID, rrr is the replication number (1 or 2), 
        and aaa is the action class label.

        Details are in: https://github.com/shahroudy/NTURGB-D 

        Args:
            filename(str): skeleton filename.        
        '''
        import re

        name = os.path.splitext(filename)[0].split()[0]
        result = list(filter(None, re.split('[a-z]+', name, flags=re.IGNORECASE)))
        return list(map(int, result))

    def _filter_data(self, item_path, settings):
        '''
        Return True to add this item and false otherwise.

        Args:
            item_path(str): path of the item.
            settings(list): camera settings, activity and subject id.

        Todo: Refactor filter.
        '''
        camera_id = settings[1]        
        replication_id = settings[3]
        activity_id = settings[4]
        if (activity_id > 49) or (replication_id != 1):
            return False # ignore all activities with 2 persons
                         # and use one replication

        frames = BodyFileReader(item_path)
        if len(frames) >= 60:
            for frame in frames:
                if len(frame) == 0:
                    return False
            return True
        return False

    def split_data(self, items):
        '''
        Split the data at random for train, eval and test set.

        Args:
            items: list of clips and their correspodning label if available.
        '''
        item_count = len(items)
        indices = np.arange(item_count)
        np.random.shuffle(indices)

        train_count = int(0.8 * item_count)
        test_count  = item_count - train_count

        train = []
        test  = []

        for i in range(train_count):
            train.append(items[indices[i]])

        for i in range(train_count, train_count + test_count):
            test.append(items[indices[i]])

        return train, test

    def write_to_csv(self, items, file_path):
        '''
        Write file path and its target pair in a CSV file format.

        Args:
            items: list of paths and their corresponding label if provided.
            file_path(str): target file path.
        '''
        if sys.version_info[0] < 3:
            with open(file_path, 'wb') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                for item in items:
                    writer.writerow(item)
        else:
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                for item in items:
                    writer.writerow(item)

    def compute_statistics(self):
        '''
        Compute some statistics across all the datatset.
        '''
        with BodyDataStatisticsPass1(self._stats_context) as stats:
            for item in self._items:
                frames = BodyFileReader(item[0])
                for frame in frames:
                    stats.add(frame[0].as_numpy())

        with BodyDataStatisticsPass2(self._stats_context) as stats:
            for item in self._items:
                frames = BodyFileReader(item[0])
                for frame in frames:
                    stats.add(frame[0].as_numpy())

        return self._stats_context

def main(input_folder, output_folder, excludeFile):
    '''
    Main entry point, it iterates through all the clip files in a folder or through all
    sub-folders into a list with their corresponding target label. It then split the data
    into training set, validation set and test set.

    Args:
        input_folder: input folder contains all the data files.
        output_folder: where to store the result.
    '''
    data_splitter = DataSplitter(input_folder, excludeFile)
    data_splitter.load_exclude_files()
    items = data_splitter.load_data_paths()

    print("{} items loaded, start splitting.".format(len(items)))

    train, test = data_splitter.split_data(items)
    print("Train: {} and test: {}.".format(len(train), len(test)))

    context = data_splitter.compute_statistics()
    print("Complete computing statistics.")

    save_statistics_context(context, os.path.join(output_folder, 'data_statistics.h5'))

    if len(train) > 0:
        data_splitter.write_to_csv(train, os.path.join(output_folder, 'train_map.csv'))
    if len(test) > 0:
        data_splitter.write_to_csv(test, os.path.join(output_folder, 'test_map.csv'))

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_folder",
                        type = str,
                        help = "Input folder containing the raw data.",
                        required = True)

    parser.add_argument("-o",
                        "--output_folder",
                        type = str,
                        help = "Output folder for the generated training and test text files.",
                        required = True)

    parser.add_argument("-e",
                        "--exclude_file",
                        type = str,
                        help = "File of skeleton files to be excluded.",
                        required = False)

    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.exclude_file)
