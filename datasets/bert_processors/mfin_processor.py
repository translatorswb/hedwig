import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class MFINProcessor(BertProcessor):
    NAME = 'MFIN'
    NUM_CLASSES = 10
    IS_MULTILABEL = False
    
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'MFIN', 'train.csv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'MFIN', 'dev.csv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'MFIN', 'test.csv')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = '%s-%s' % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples