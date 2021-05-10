import os
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import pydicom

#Dataset input folder (original files downloaded from CBIS-DDSM web site, Ex: Mass-Training Full Mammogram Images (DICOM), Mass-Test Full Mammogram Images (DICOM))
raw_dataset_root_path = '/media/rogerio/49a45a42-45ee-49c3-9085-9defff31abe8/TCC-Dataset/dataset/raw/'
#Dataset output folder (pre processed files after script execution)
converted_dataset_root_path = '/media/rogerio/49a45a42-45ee-49c3-9085-9defff31abe8/TCC-Dataset/dataset/converted/'
subset_path = ['Mass-Training-Full-Mammogram-Images-(DICOM)', 'Mass-Test-Mammogram-Images-(DICOM)']
subset_metadata_path =  ['mass_case_description_train_set.csv', 'mass_case_description_test_set.csv']
classification_subset_file = ['model_train_dataset_true_classification', 'model_test_dataset_true_classification']

def mass_training_handler(path_file):
    return 'mass-training' in path_file.lower() and not 'cropped' in path_file.lower()

def mass_training_roi_handler(path_file):
    return 'mass-training' in path_file.lower() and 'cropped' in path_file.lower()

def mass_test_handler(path_file):
    return 'mass-test' in path_file.lower() and not 'cropped' in path_file.lower()

def calc_training_handler(path_file):
    return 'calc-training' in path_file.lower() and not 'cropped' in path_file.lower()

def calc_training_roi_handler(path_file):
    return 'calc-training' in path_file.lower() and 'cropped' in path_file.lower()    

dataset_category = {
    'mass-training': {
        'handler': mass_training_handler,
        'files': []
    },
    'mass-training-roi': {
        'handler': mass_training_roi_handler,
        'files': []
    },
    'mass-training-roi': {
        'handler': mass_test_handler,
        'files': []
    },
    'calc-training': {
        'handler': calc_training_handler,
        'files': []
    },
    'calc-training-roi': {
        'handler': calc_training_roi_handler,
        'files': []
    }
}

abnormality_mapper = {
    'BENIGN': 0,
    'BENIGN_WITHOUT_CALLBACK': 1,
    'MALIGNANT': 2
}

average_image_size = (300, 300) #(Width, Height)

def set_data_category(path):
    for category in dataset_category.keys():
        result = dataset_category[category]['handler'](path)
        if result: 
            dataset_category[category]['files'].append(path)
            break

def walk_over_directory_tree(path):
    for root_folder, subfolders, files in os.walk(path):
        if(len(files)>1):
            for _file in files:
                set_data_category('{path}/{file}'.format(path=root_folder,file=_file))
        if (len(files) == 1):
            if(len(subfolders) == 0):
                set_data_category('{path}/{file}'.format(path=root_folder,file=files[0]))

def read_images(subset_path, subset_metadata_path, classification):
    malignant_counter = 0
    for category in [category for category in dataset_category.keys() if len(dataset_category[category]['files']) > 0]:
        
        print("================= Reading data from {category} category of dataset ====================".format(category=category.upper()))

        for _file in dataset_category[category]['files']:
            
            dataset_element = pydicom.dcmread(_file)
            image = dataset_element.pixel_array
            dataset_element_metadata = dataset_element.PatientID.split('_') 
            
            if(len(dataset_element_metadata) == 4):
                patient_prefix, patient_code, breast_side, exam_type = dataset_element_metadata
                element_id = ''
                training_type = 'Mass-Test'
            if(len(dataset_element_metadata) == 5):
                training_type, patient_prefix, patient_code, breast_side, exam_type = dataset_element_metadata
                element_id = ''
            else:
                training_type, patient_prefix, patient_code, breast_side, exam_type, element_id = dataset_element_metadata

            patient_id = "{prefix}_{code}".format(prefix=patient_prefix,code=patient_code)
            
            print("\nTraining ................:", training_type)
            print("Patient's Id ............:", patient_id)
            print("Patient's Exam ..........:", exam_type)
            print("Patient's Breat Side ....:", breast_side)
            print("Image Size ..............: Width: {0} x Height: {1}".format(image.shape[1], image.shape[0]))

            print(".............. Resizing image to {0} x {1} ..............".format(average_image_size[0], average_image_size[1]))
            resized_image = cv.resize(image, average_image_size, 0, 0, interpolation=cv.INTER_AREA)
            
            print(".............. Getting image metadata ..............")
            subset_metadata = pd.read_csv("{dataset}{subset_metadata}".format(dataset=raw_dataset_root_path,subset_metadata=subset_metadata_path))
            
            print(".............. Getting classification of the abnormality  ..............")
            image_metadata = subset_metadata[
                (subset_metadata['patient_id'] == patient_id) &
                (subset_metadata['left or right breast'] == breast_side) &
                (subset_metadata['image view'] == exam_type)
            ]
            
            if image_metadata['pathology'].values[0] == 'BENIGN' or image_metadata['pathology'].values[0] == 'MALIGNANT':
                if not os.path.exists('{0}/{1}/{2}'.format(converted_dataset_root_path, subset_path, image_metadata['pathology'].values[0])):
                    os.makedirs('{0}/{1}/{2}'.format(converted_dataset_root_path, subset_path, image_metadata['pathology'].values[0]))

            if image_metadata['pathology'].values[0] == 'BENIGN':
                classification.append(abnormality_mapper['BENIGN'])
            elif image_metadata['pathology'].values[0] == 'MALIGNANT':  
                classification.append(abnormality_mapper['BENIGN_WITHOUT_CALLBACK'])
            else:
                continue
            
            filename = '{0}.png'.format(dataset_element.PatientID)
            cv.imwrite('{0}/{1}/{2}/{3}'.format(converted_dataset_root_path, subset_path, image_metadata['pathology'].values[0], filename), resized_image)
            
            # -------------------------------- Normalizando quantidade de imagens no dataset para as duas classes ---------------------------------
            if subset_path == 'Mass-Test-Mammogram-Images-(DICOM)' and image_metadata['pathology'].values[0] == 'MALIGNANT' and malignant_counter < 41:
                filename = '{0}_copy.png'.format(dataset_element.PatientID)
                cv.imwrite('{0}/{1}/{2}/{3}'.format(converted_dataset_root_path, subset_path, image_metadata['pathology'].values[0], filename), resized_image)
                classification.append(1)
                malignant_counter += 1


    dataset_category[category]['files'] = []

for index in range(len(subset_path)):
    print('.......... Pre processing {} dataset .......... \n'.format(subset_path[index]))
    classification = []
    walk_over_directory_tree("{dataset}{subset}".format(dataset=raw_dataset_root_path,subset=subset_path[index]))
    read_images(subset_path[index], subset_metadata_path[index], classification)
    print(".............. Creating {} picke file ..............".format(classification_subset_file[index]))
    pickle_out = open(classification_subset_file[index], "wb")
    pickle.dump(np.array(classification).astype('int64'), pickle_out)
    pickle_out.close()
    print(".............. File {} created ..............".format(classification_subset_file[index]))
    print('.......... {} Dataset has {} items .......... \n\n'.format(subset_path[index], len(classification)))
