import warnings

warnings.filterwarnings('ignore')
import os
from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd


class GQADataset(Dataset):
    def __init__(
            self,
            image_dir,
            annotations,
    ):
        self.data = pd.read_csv(annotations)
        self.src = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.src, row['image'])
        return Image.open(path).convert('RGB'), row.question, row.answer

    @staticmethod
    def gen_csv(
            image_dir,
            annotations,
            istrain,
            dst
    ):
        data = {
            'image': [],
            'question': [],
            'answer': [],
        }
        if istrain:
            for file in os.listdir(annotations):
                ann = json.load(open(os.path.join(annotations, file)))
                for qid in ann:
                    question = ann[qid]
                    data['image'].append(os.path.join(image_dir, question['imageId'] + '.jpg'))
                    data['question'].append(question['question'])
                    data['answer'].append(question['fullAnswer'])
        else:
            ann = json.load(open(annotations))
            for qid in ann:
                question = ann[qid]
                data['image'].append(os.path.join(image_dir, question['imageId'] + '.jpg'))
                data['question'].append(question['question'])
                data['answer'].append(question['fullAnswer'])
        data = pd.DataFrame(data)
        data.to_csv(dst, index=False)


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
                    [
                        os.path.join(self.image_src, split, f'{image_id:012d}.jpg'),
                        question,
                        ans['answer'],
                    ]
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
                    [
                        os.path.join(self.image_src, split, f'{image_id:012d}.jpg'),
                        question,
                        ans['answer'],
                    ]
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


if __name__ == '__main__':
    gqa_train = '/data/gqa/annotations/train_all_questions'
    gqa_val = '/data/gqa/annotations/val_all_questions.json'
    gqa_src = '/data/gqa/images'
    GQADataset.gen_csv(
        gqa_src,
        gqa_train,
        True,
        '/data/gqa/annotations/train.csv'
    )
    GQADataset.gen_csv(
        gqa_src,
        gqa_val,
        False,
        '/data/gqa/annotations/val.csv'
    )
