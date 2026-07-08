import bisect
import functools
import glob
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)


def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset

def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train == True:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False):
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
  _path = os.path.join(cache_dir, filename)
  
  if utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2
  
  if dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only', cache_dir=cache_dir)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming)

  if dataset_name in ['lambada', 'openwebtext-train',
                      'openwebtext-valid']:
    data = dataset
  else:
    data = dataset[mode]

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith('scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    else:
      text = example['text']
    
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  if dataset_name == 'ptb':
    tokenized_dataset = tokenized_dataset.remove_columns(
      'sentence')
  elif 'scientific_papers' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns([
      'article', 'abstract', 'section_names'])
  elif dataset_name == 'ag_news':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'label'])
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is not None:
      tokenizer.bos_token = tokenizer.cls_token
    elif tokenizer.eos_token is not None:
      # e.g. T5, which has neither a bos_token nor a cls_token.
      tokenizer.bos_token = tokenizer.eos_token
    else:
      raise AttributeError(
        'Tokenizer must have a bos_token, cls_token, '
        f'or eos_token: {tokenizer}')
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer
    

_SHARD_METADATA_FILE = '_shard_metadata.json'


class PackedArrowShardDataset(torch.utils.data.Dataset):
  """Map-style dataset over pre-tokenized, pre-packed Arrow IPC shard files.

  These shards are produced by
  ``sde_llm/scripts/prepare_pretrain_data.py`` -- the exact corpus the AR-M
  baseline pretrains on. Every row already holds ``model.length`` token ids
  (T5-small tokenized, packed with no padding) in the ``target_ids`` column,
  so no tokenization / grouping happens here.

  Shards are memory-mapped and opened lazily inside each DataLoader worker, so
  token data is served from the OS page cache and never copied into the Python
  heap. Because pyarrow memory-maps are not fork-safe to share, the per-shard
  table cache is keyed by pid and rebuilt in each worker.
  """

  ID_COLUMN = 'target_ids'

  def __init__(self, shard_paths):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    if not shard_paths:
      raise ValueError(
        'PackedArrowShardDataset received an empty list of shards.')
    self.shard_paths = list(shard_paths)
    # Row counts come from the IPC footers only; read_all() is mmap-backed and
    # does not copy the token data.
    self._shard_lens = []
    for path in self.shard_paths:
      with pa.memory_map(path, 'r') as mm:
        self._shard_lens.append(ipc.open_file(mm).read_all().num_rows)
    self._offsets = [0]
    for n in self._shard_lens:
      self._offsets.append(self._offsets[-1] + n)
    self._length = self._offsets[-1]
    self._tables = None   # {shard_idx: (mmap, table)} — populated per worker
    self._pid = None

  def __len__(self):
    return self._length

  def _get_table(self, shard_idx):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    pid = os.getpid()
    if self._tables is None or self._pid != pid:
      self._tables = {}
      self._pid = pid
    cached = self._tables.get(shard_idx)
    if cached is None:
      mm = pa.memory_map(self.shard_paths[shard_idx], 'r')
      table = ipc.open_file(mm).read_all()
      # Keep the mmap referenced so the table's buffers stay valid.
      self._tables[shard_idx] = (mm, table)
      return table
    return cached[1]

  def _locate(self, idx):
    if idx < 0:
      idx += self._length
    shard_idx = bisect.bisect_right(self._offsets, idx) - 1
    return shard_idx, idx - self._offsets[shard_idx]

  def __getitem__(self, idx):
    shard_idx, local_idx = self._locate(idx)
    ids = self._get_table(shard_idx).column(self.ID_COLUMN)[local_idx].as_py()
    return {'input_ids': torch.tensor(ids, dtype=torch.long)}


def _to_instruction_example(condition_ids, target_ids,
                            seq_length, pad_token_id):
  """Builds a fixed-length instruction-tuning example.

  Mirrors the AR-M IT collation (sde_llm data_utils.get_dataloader with
  pad_token: eos): input = condition + target, right-padded to
  ``seq_length`` with EOS so the model learns to terminate responses.
  ``cond_mask`` marks the condition positions -- these are never noised
  by the forward process and never scored; the response and the trailing
  EOS padding are.

  Examples are pre-filtered to condition + target <= seq_length by
  prepare_instruction_data.py; truncation here is a defensive no-op.
  """
  seq = list(condition_ids) + list(target_ids)
  seq = seq[:seq_length]
  cond_len = min(len(condition_ids), seq_length)
  input_ids = torch.full(
    (seq_length,), pad_token_id, dtype=torch.long)
  input_ids[:len(seq)] = torch.tensor(seq, dtype=torch.long)
  cond_mask = torch.zeros(seq_length, dtype=torch.bool)
  cond_mask[:cond_len] = True
  return {'input_ids': input_ids, 'cond_mask': cond_mask}


class InstructionShardDataset(PackedArrowShardDataset):
  """Instruction-tuning pairs from Arrow IPC shards.

  Shards are produced by ``sde_llm/scripts/prepare_instruction_data.py``
  (smol-smoltalk / ultrachat_200k / tulu-3-sft, T5-small tokenized): each
  row holds ``condition_ids`` (chat history up to and including the
  trailing "[assistant]" tag) and ``target_ids`` (the final assistant
  response). Inherits the lazy per-worker mmap machinery from
  ``PackedArrowShardDataset``.
  """

  def __init__(self, shard_paths, seq_length, pad_token_id):
    super().__init__(shard_paths)
    self.seq_length = seq_length
    self.pad_token_id = pad_token_id

  def __getitem__(self, idx):
    shard_idx, local_idx = self._locate(idx)
    table = self._get_table(shard_idx)
    condition_ids = table.column('condition_ids')[local_idx].as_py()
    target_ids = table.column('target_ids')[local_idx].as_py()
    return _to_instruction_example(
      condition_ids, target_ids, self.seq_length, self.pad_token_id)


class InstructionValidDataset(torch.utils.data.Dataset):
  """Same example format as ``InstructionShardDataset``, over the valid
  split written by prepare_instruction_data.py (HF save_to_disk layout).

  The .arrow file is read directly with pyarrow rather than
  ``datasets.load_from_disk`` -- the split was written by a newer
  `datasets` version whose feature schema this env cannot parse. The
  split is small (~1000 rows), so it is materialized eagerly, which also
  keeps it fork-safe for DataLoader workers.
  """

  def __init__(self, dataset_path, seq_length, pad_token_id):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    arrow_paths = sorted(
      glob.glob(os.path.join(dataset_path, '*.arrow')))
    if not arrow_paths:
      raise FileNotFoundError(
        f'No .arrow files found in {dataset_path!r}.')
    conditions, targets = [], []
    for path in arrow_paths:
      with pa.memory_map(path, 'r') as mm:
        try:
          table = ipc.open_stream(mm).read_all()
        except pa.lib.ArrowInvalid:
          table = ipc.open_file(mm).read_all()
        conditions.extend(table.column('condition_ids').to_pylist())
        targets.extend(table.column('target_ids').to_pylist())
    self.examples = list(zip(conditions, targets))
    self.seq_length = seq_length
    self.pad_token_id = pad_token_id

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    condition_ids, target_ids = self.examples[idx]
    return _to_instruction_example(
      condition_ids, target_ids, self.seq_length, self.pad_token_id)


def get_instruction_datasets(config, tokenizer):
  """Builds (train, valid) instruction-tuning datasets shared with the
  AR-M baseline's IT stage."""
  train_dir = config.data.instruction_train_dir
  meta_path = os.path.join(train_dir, _SHARD_METADATA_FILE)
  if not os.path.isfile(meta_path):
    raise FileNotFoundError(
      f'No {_SHARD_METADATA_FILE} found in {train_dir!r}; expected a '
      'directory of Arrow IPC shards from prepare_instruction_data.py.')
  shard_paths = sorted(
    glob.glob(os.path.join(train_dir, 'shard_*.arrow')))
  if not shard_paths:
    raise FileNotFoundError(f'No shard_*.arrow files in {train_dir!r}.')
  # EOS padding, mirroring the AR-M IT recipe (pad_token: eos).
  pad_token_id = tokenizer.eos_token_id
  train_set = InstructionShardDataset(
    shard_paths, config.model.length, pad_token_id)
  valid_set = InstructionValidDataset(
    config.data.instruction_valid_dir, config.model.length,
    pad_token_id)
  LOGGER.info(
    f'Instruction corpus: {len(train_set)} train examples from '
    f'{len(shard_paths)} shards, {len(valid_set)} valid examples.')
  return train_set, valid_set


def get_packed_pretrain_datasets(pretrain_dir, n_valid_shards=1):
  """Build (train, valid) datasets from a directory of Arrow IPC shards.

  The last ``n_valid_shards`` shard(s) are held out for validation (the AR-M
  baseline uses no held-out split, so this is a small, negligible carve-out
  purely to give MDLM's Lightning loop a validation signal).
  """
  meta_path = os.path.join(pretrain_dir, _SHARD_METADATA_FILE)
  if not os.path.isfile(meta_path):
    raise FileNotFoundError(
      f'No {_SHARD_METADATA_FILE} found in {pretrain_dir!r}; expected a '
      'directory of Arrow IPC shards from prepare_pretrain_data.py.')
  shard_paths = sorted(
    glob.glob(os.path.join(pretrain_dir, 'shard_*.arrow')))
  if len(shard_paths) <= n_valid_shards:
    raise ValueError(
      f'Found {len(shard_paths)} shard(s) in {pretrain_dir!r}, need more '
      f'than n_valid_shards={n_valid_shards}.')
  train_paths = shard_paths[:-n_valid_shards]
  valid_paths = shard_paths[-n_valid_shards:]
  LOGGER.info(
    f'Packed pretrain corpus: {len(train_paths)} train shards, '
    f'{len(valid_paths)} valid shard(s) from {pretrain_dir}')
  return (PackedArrowShardDataset(train_paths),
          PackedArrowShardDataset(valid_paths))


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  # Pre-tokenized / pre-packed corpus (shared with the AR-M baseline): the
  # rows are already token-id windows of length model.length, so we bypass the
  # HuggingFace tokenize/group pipeline entirely.
  packed_pretrain = config.data.train == 'pretrain'
  if packed_pretrain:
    packed_train_set, packed_valid_set = get_packed_pretrain_datasets(
      config.data.pretrain_dir,
      n_valid_shards=config.data.get('n_valid_shards', 1))
  # Instruction-tuning pairs (shared with the AR-M baseline's IT stage):
  # like the packed corpus, rows are pre-tokenized, so we bypass the
  # HuggingFace tokenize/group pipeline.
  instruction = config.data.train == 'instruction'
  if instruction:
    instruction_train_set, instruction_valid_set = (
      get_instruction_datasets(config, tokenizer))

  if skip_train:
    train_set = None
  elif packed_pretrain:
    train_set = packed_train_set
  elif instruction:
    train_set = instruction_train_set
  else:
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length)

  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  elif packed_pretrain:
    valid_set = packed_valid_set
  elif instruction:
    valid_set = instruction_valid_set
  else:
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=False)

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      persistent_workers=True)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0