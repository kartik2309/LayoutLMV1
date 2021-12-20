import copy
import json
import os
import pandas as pd
from transformers import AutoTokenizer


class DataPreparation:
    def __init__(self, data_folder, max_samples=None):
        train_data_path = os.path.join(data_folder, 'training_data')
        test_data_path = os.path.join(data_folder, 'testing_data')

        train_images_path = os.path.join(train_data_path, 'images')
        train_annotations_path = os.path.join(train_data_path, 'annotations')
        train_images_list = os.listdir(train_images_path)

        test_images_path = os.path.join(test_data_path, 'images')
        test_annotations_path = os.path.join(test_data_path, 'annotations')
        test_images_list = os.listdir(test_images_path)

        if max_samples is not None:
            train_images_list = train_images_list[:max_samples]
            test_images_list = test_images_list[:max_samples]

        self.train_annotations = {}
        for image_name in train_images_list:
            imageid = image_name.strip('.png')
            f_path = os.path.join(train_annotations_path, imageid + '.json')
            with open(f_path) as f:
                f_dict = json.load(f)
                self.train_annotations[imageid] = copy.deepcopy(f_dict)
                f.close()

        self.test_annotations = {}
        for image_name in test_images_list:
            imageid = image_name.strip('.png')
            f_name = os.path.join(test_annotations_path, imageid + '.json')
            with open(f_name) as f:
                f_dict = json.load(f)
                self.test_annotations[imageid] = copy.deepcopy(f_dict)
                f.close()

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')

        self.label_tags_no_iob = {
            'other': 1,
            'header': 2,
            'question': 3,
            'answer': 4
        }

        return

    def __prepare_dataframe_from_dict(self, input_dict):
        imageids = []
        bboxes = []
        texts = []
        labels = []

        bbox = []
        text = []
        label = []
        for key, value in input_dict.items():
            imageids.append(key)
            forms = value['form']

            bbox.clear()
            text.clear()
            label.clear()

            for form in forms:
                if form['text'] == "":
                    continue
                words = form['words']
                for word in words:
                    bbox.append(word['box'])
                    text.append(word['text'])
                label.append(
                    self.label_tags_no_iob[form['label']]
                )

            bboxes.append(copy.deepcopy(bbox))
            texts.append(' '.join(text))
            labels.append(copy.deepcopy(label))

        tokenized = self.tokenizer.batch_encode_plus(texts, max_length=512, padding='max_length', truncation=True)

        input_ids = tokenized['input_ids']
        token_type_ids = tokenized['token_type_ids']
        attention_mask = tokenized['attention_mask']

        df = pd.DataFrame({
            'imageid': imageids,
            'bboxes': bboxes,
            'texts': texts,
            'labels': labels,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        })

        return df

    def prepare_dataframe(self):
        df_train = self.__prepare_dataframe_from_dict(input_dict=self.train_annotations)
        df_test = self.__prepare_dataframe_from_dict(input_dict=self.test_annotations)

        return df_train, df_test










