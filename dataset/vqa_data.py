import warnings
warnings.filterwarnings('ignore')
import os
from torch.utils.data import Dataset
from PIL import Image
import json


class VQAv2Datasets:
    def __init__(
        self,
        instances2017,
        qa_anns2014,
        qa_questions2014,
        image_src,
    ):
        self.images = {}
        for split, instance in zip(['train2017', 'val2017'], instances2017):
            instance = json.load(open(instance))
            for image in instance['images']:
                self.images[image['id']] = split
        self.qa_anns2014 = [json.load(open(ann)) for ann in qa_anns2014]
        self.qa_questions2014 = qa_questions2014
        self.image_src = image_src
        self.questions = {}
        for ques in qa_questions2014:
            ques = json.load(open(ques))
            for q in ques['questions']:
                self.questions[q['question_id']] = q

    def train2014(self):
        data = []
        for ann in self.qa_anns2014[0]['annotations']:
            question_id = ann['question_id']
            question = self.questions[question_id]['question']
            image_id = ann['image_id']
            split = self.images[image_id]
            for ans in ann['answers']:
                # if not os.path.exists(os.path.join(self.image_src, split, f'{image_id:012d}.jpg')):
                #     print(os.path.join(split, f'{image_id:012d}.jpg'))
                #     continue
                data.append(
                    [os.path.join(self.image_src, split, f'{image_id:012d}.jpg'),
                    question,
                    ans['answer'],]
                )
        return VQAv2Dataset(data)

    def val2014(self):
        data = []
        for ann in self.qa_anns2014[1]['annotations']:
            question_id = ann['question_id']
            question = self.questions[question_id]['question']
            image_id = ann['image_id']
            split = self.images[image_id]
            for ans in ann['answers']:
                # if not os.path.exists(os.path.join(self.image_src, split, f'{image_id:012d}.jpg')):
                #     print(os.path.join(split, f'{image_id:012d}.jpg'))
                #     continue
                data.append(
                    [os.path.join(self.image_src, split, f'{image_id:012d}.jpg'),
                    question,
                    ans['answer'],]
                )
        return VQAv2Dataset(data)


class VQAv2Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, question, answer = self.data[i]
        image = Image.open(image).convert('RGB')
        return image, question, answer
