# class CustomDataset(Dataset):
#     def __init__(self, dataframe, tokenizer, source_len, summ_len):
#         self.tokenizer = tokenizer
#         self.data = dataframe
#         self.source_len = source_len
#         self.summ_len = summ_len
#         self.text = self.data.text
#         self.ctext = self.data.ctext

#     def __len__(self):
#         return len(self.text)

#     def __getitem__(self, index):
#         ctext = str(self.ctext[index])
#         ctext = ' '.join(ctext.split())

#         text = str(self.text[index])
#         text = ' '.join(text.split())

#         source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt', truncation = True)
#         target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt', truncation = True)

#         source_ids = source['input_ids'].squeeze()
#         source_mask = source['attention_mask'].squeeze()
#         target_ids = target['input_ids'].squeeze()
#         target_mask = target['attention_mask'].squeeze()
	
# 	y_ids = target_ids[:, :-1].contiguous()
#         lm_labels = target_ids[:, 1:].clone().detach()
#         lm_labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100


#         return {
#             'source_ids': source_ids.to(dtype=torch.long), 
#             'source_mask': source_mask.to(dtype=torch.long), 
#             'target_ids': y_ids.to(dtype=torch.long),
#             'lm_labels': lm_labels.to(dtype=torch.long)
#         }
import copy
import json
import logging
import os

import torch
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
    """

    def __init__(self, guid, src_text, tgt_text):
        self.guid = guid
        self.src_text = src_text
        self.tgt_text = tgt_text
    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, source_ids, source_mask, target_ids, target_ids_y):
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_ids_y = target_ids_y

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Processor(object):
    """Processor for the data set """

    def __init__(self, args):
        self.args = args
        
        self.src_text_file = "seq.in"
        self.tgt_text_file = "seq.out"

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads line by line of a file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, src_texts, tgt_texts, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (src_text, tgt_text) in enumerate(zip(src_texts, tgt_texts)):
            guid = "%s-%s" % (set_type, i)
            # 1. source_text
            source = src_text  # Some are spaced twice
            # 3. target_text
            target = tgt_text
            examples.append(InputExample(guid=guid, src_text=source, tgt_text=target))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            src_texts=self._read_file(os.path.join(data_path, self.src_text_file)),
            tgt_texts=self._read_file(os.path.join(data_path, self.tgt_text_file)),
            set_type=mode
        )


def convert_examples_to_features(
    examples,
    max_source_len,
    max_target_len,
    tokenizer,
    pad_token_label_id=-100,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        source = example.src_text
        target = example.tgt_text

        source = tokenizer.batch_encode_plus([source], max_length= max_source_len, padding = 'max_length', return_tensors='pt', truncation = True)
        target = tokenizer.batch_encode_plus([target], max_length= max_target_len, padding = 'max_length', return_tensors='pt', truncation = True)

        source_ids = source['input_ids'].squeeze().tolist()
        source_mask = source['attention_mask'].squeeze().tolist()
        target_ids = target['input_ids'].squeeze().tolist()
        target_mask = target['attention_mask'].squeeze().tolist()

        # y_ids = target_ids[:, :-1].contiguous()
        # lm_labels = target_ids[:, 1:].clone().detach()
        # lm_labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("source: %s" % source)
            logger.info("src_input_ids: %s" % source_ids)
            logger.info("src_attention_mask: %s" % source_mask)
            logger.info("target: %s" % target)
            logger.info("tgt_input_ids: %s" % target_ids)
            logger.info("tgt_attention_mask: %s" % target_mask)

        features.append(
            InputFeatures(
            source_ids = source_ids, 
            source_mask = source_mask, 
            target_ids = target_ids,
            target_ids_y = target_ids
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = Processor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_source_len, args.max_target_len
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(
            examples, args.max_source_len,args.max_target_len, tokenizer, pad_token_label_id=pad_token_label_id
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    # print(features[0].source_ids)
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    all_target_ids_y = torch.tensor([f.target_ids_y for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_source_ids, all_source_mask, all_target_ids, all_target_ids_y
    )
    return dataset
