import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torch import tensor


class FUNSDDataset(Dataset):
    def __init__(self, image_data_folder, input_ids, token_type_ids, attention_mask,
                 imageid, bbox, image_size=1000, labels=None):
        self.image_data_folder = image_data_folder
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.imageid = imageid
        self.bbox = bbox
        self.image_size = image_size

        if labels is not None:
            self.labels = labels

    def __get_image(self, imageid):
        image_path = os.path.join(self.image_data_folder, imageid + '.png')
        pil_image = Image.open(image_path).convert('RGB')
        width, height = pil_image.size
        pil_image = pil_image.resize((self.image_size, self.image_size))
        image_tensor = to_tensor(pil_image)
        return image_tensor, width, height

    def __resize_bbox_for_layoutlm(self, bbox, width, height):
        return (
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height))
        )

    def __resize_bbox(self, bbox, width, height):
        return (
            self.image_size * (bbox[0] / width),
            self.image_size * (bbox[1] / height),
            self.image_size * (bbox[2] / width),
            self.image_size * (bbox[3] / height)
        )

    def __pad_list(self, input_list, value_to_pad, max_len=512):
        len_input_list = len(input_list)
        if len_input_list < max_len:
            diff = max_len - len_input_list
            to_extend = [value_to_pad] * diff
            input_list.extend(to_extend)
            return input_list, len_input_list

        elif len_input_list > max_len:
            return input_list[:max_len], max_len

        else:
            return input_list, max_len

    def __getitem__(self, idx):
        item = {'input_ids': tensor(self.input_ids[idx]), 'token_type_ids': tensor(self.token_type_ids[idx]),
                'attention_mask': tensor(self.attention_mask[idx]), 'imageid': self.imageid[idx]}

        item['images'], width, height = self.__get_image(self.imageid[idx])
        bbox_rcnn = [self.__resize_bbox(bbox_, width, height) for bbox_ in self.bbox[idx]]
        bbox_rcnn, orig_len = self.__pad_list(bbox_rcnn, value_to_pad=(0, 0, 0, 0))
        item['bbox_rcnn'] = tensor(bbox_rcnn)
        item['orig_len'] = orig_len

        bbox_layoutlm = [self.__resize_bbox_for_layoutlm(bbox_, width, height) for bbox_ in self.bbox[idx]]
        bbox_layoutlm, _ = self.__pad_list(bbox_layoutlm, value_to_pad=(0, 0, 0, 0))
        item['bbox_layoutlm'] = tensor(bbox_layoutlm)

        if 'labels' in self.__dict__:
            labels = self.labels[idx]
            labels, _ = self.__pad_list(labels, value_to_pad=0)

            item['labels'] = tensor(labels)

        return item

    def __len__(self):
        return len(self.labels)
