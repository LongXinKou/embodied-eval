import tensorflow as tf
import re
import random

from IPython import display
from PIL import Image
import matplotlib.pyplot as plt

def as_gif(images):
  # Render the images as the gif:
  images[0].save('/tmp/temp.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
  gif_bytes = open('/tmp/temp.gif','rb').read()
  return gif_bytes

def display_image(images):
  if len(images) >1 :
    result_images = []
    for x in images:
      result_images.append(Image.fromarray(x))
    display.display(display.Image(as_gif(result_images)))
  else:
    plt_image =Image.fromarray(images[0])
    plt.imshow(plt_image)
    plt.show()
    
class Task:
  """A class for handling tags and splits in a given task."""

  # Tags for default splitting, based on who is talking.
  PRED_STARTS = ['Robot:', 'Thought:', 'Action:']
  NOPRED_STARTS = ['User:', 'System:']

  # Tags surrounding all blocks needing to be predicted by the model.
  PRED_START = '<PRED>'
  PRED_END = '</PRED>'
  # Tags surrounding only binary answers, typically 'yes' and 'no'.
  PRED_ANSWER_BINARY_START = '<PRED:ANSWER:BINARY>'
  PRED_ANSWER_BINARY_END = '</PRED:ANSWER:BINARY>'
  # Tags surrounding all discrete answers coming from a limited set of classes,
  # e.g. 'yes', 'no', 'halfway there', 'done', '10s', etc.
  PRED_ANSWER_DISCRETE_START = '<PRED:ANSWER:DISCRETE>'
  PRED_ANSWER_DISCRETE_END = '</PRED:ANSWER:DISCRETE>'
  # Tags surrounding things that constitute an answer to a question,
  # the question may be asked by a user or by the model itself.
  PRED_ANSWER_START = '<PRED:ANSWER'
  PRED_ANSWER_END = '</PRED:ANSWER'
  # Tags that have any sort of short-content value
  PRED_ALL_START = '<PRED:'
  PRED_ALL_END = '</PRED:'

  TAGS_RE = r'(</*\w[:\w]*>)'

  def __init__(self, text):
    self.text = text

  def get_random_split(self, split_type='speaker'):
    splits = self.get_splits(split_type)
    return random.choice(splits)

  def get_splits(self, split_type='speaker'):
    """Returns a list of (source, target) split pairs."""
    if split_type == 'pred':
      return self.get_splits_from_tags(
          start_tags=[self.PRED_START], end_tags=[self.PRED_END])
    elif split_type == 'binary':
      return self.get_splits_from_tags(
          start_tags=[self.PRED_BINARY_START], end_tags=[self.PRED_BINARY_END])
    elif split_type == 'discrete':
      return self.get_splits_from_tags(
          start_tags=[self.PRED_DISCRETE_START],
          end_tags=[self.PRED_DISCRETE_END])
    elif split_type == 'answer':
      return self.get_splits_from_tags(
          start_tags=[self.PRED_ANSWER_START], end_tags=[self.PRED_ANSWER_END])
    elif split_type == 'A:':
      return self.get_splits_from_tags(start_tags=['A:'], end_tags=[])
    elif split_type == 'speaker':
      return self.get_splits_from_tags(
          start_tags=self.PRED_STARTS, end_tags=self.NOPRED_STARTS)
    elif split_type == 'all':
      return self.get_splits_from_tags(
          start_tags=[self.PRED_ALL_START], end_tags=[self.PRED_ALL_END]
      )
    else:
      raise ValueError('Unknown split type: %s' % split_type)

  def get_splits_from_tags(self, start_tags, end_tags):
    """Returns a list of (source, target) split pairs given start/end tags."""
    # Find all the first positions of a start element.
    split_positions = []
    position = 0
    while position < len(self.text):
      # Find the next start tag given current position.
      start_position = self.find_next_tag(position, start_tags)
      if start_position is None:
        break
      # Then find the first end tag after this start tag.
      end_position = self.find_next_tag(start_position, end_tags)
      if end_position is None:
        end_position = len(self.text)
      split_positions.append((start_position, end_position))
      position = end_position + 1
    return self.get_splits_from_positions(split_positions)

  def get_splits_from_positions(self, split_positions):
    """Returns a list of (source, target) split pairs given split positions."""
    # Create splits.
    splits = []
    for (split_position, end_position) in split_positions:
      source = ''
      if split_position > 0:
        source = self.text[:split_position]
        source = self._remove_tags(source)
      target = self.text[split_position:end_position]
      target = self._remove_tags(target)
      splits.append((source, target))

    # If no splits are found, return entire text.
    if not splits:
      splits = [('', self.text)]

    return splits

  def find_next_tag(self, position, tags):
    tag_position = None
    lower_text = self.text.lower()
    for tag in tags:
      p = lower_text.find(tag.lower(), position)
      if p >= 0 and (tag_position is None or p < tag_position):
        tag_position = p
    return tag_position

  def _remove_tags(self, text):
    return re.sub(self.TAGS_RE, '', text)

  def remove_tags(self):
    self.text = self._remove_tags(self.text)

  def __str__(self):
    return self.text
  
class Tasks():
  """A class for handling and holding tasks information."""

  TASK_RE = r'(<task[:\w]*>)'
  RE_FLAGS = re.IGNORECASE

  def __init__(self, tasks_raw=None):
    # Contains all tasks for each task type in this Tasks collection
    # key: str, task type (tag)
    # value: list[str], question-answers which belong to this task type
    self.tasks_dict = {}
    self.tasks_list = []
    self.tasks_types = []
    self.tasks_raw = tasks_raw
    if tasks_raw is not None:
      self.add(tasks_raw)

  def add(self, tasks):
    self.add_from_text(tasks)

  def add_from_dict(self, tasks_dict):
    for name, tasks in tasks_dict.items():
      if name not in self.tasks_dict:
        self.tasks_dict[name] = []
      self.tasks_dict[name].extend(tasks)
      self.tasks_list.extend(tasks)
      self.tasks_types.extend([name] * len(tasks))

  def add_from_text(self, text):
    task_dict = self.text_to_dict(text)
    self.add_from_dict(task_dict)

  def text_to_dict(self, text):
    """Returns all tasks associated with this video."""
    # Split a serialized string into raw strings of individual tasks
    split = re.split(self.TASK_RE, text, flags=self.RE_FLAGS)[1:]
    # Construct a dict of
    # key: str, task type (tag)
    # value: list[str], question-answers which belong to this task type
    tasks_dict = {}
    i = 0
    while i < len(split) - 1:
      tag = split[i].strip()
      task = split[i+1].lstrip()
      if task:
        if tag not in tasks_dict:
          tasks_dict[tag] = []
        tasks_dict[tag].append(task)
      i += 2
    return tasks_dict

  def __str__(self, show_tasks=True):
    s_parts = []
    s_parts.append('%d task types in %d tasks:\n' % (
        len(self.tasks_dict.keys()), len(self)))
    for key in sorted(self.tasks_dict):
      tasks = self.tasks_dict[key]
      s_parts.append('%s (%d / %d, %.1f%%)' % (
          key, len(tasks), len(self), 100 * len(tasks) / float(len(self))))
      if show_tasks:
        s_parts.append('\n\t%s' % str(tasks))
      s_parts.append('\n')
    return ''.join(s_parts)

  def detailed_str(self):
    s_parts = []
    s_parts.append('Raw input: %s' % str(self.tasks_raw))
    s_parts.append('\n%s' % self.__str__(show_tasks=True))
    return ''.join(s_parts)

  def get_stats(self):
    return self.__str__(show_tasks=False)

  def __len__(self):
    return len(self.tasks_list)

  def get_tasks_list(self):
    return self.tasks_list

  def get_tasks_types(self):
    return self.tasks_types

  def get_random_task(self):
    if not self.tasks_list:
      raise ValueError('Unexpected empty tasks list')
    return random.choice(self.tasks_list)

  def sample_task(self, weights):
    """Sample a task using weights associated with task patterns.

    Note: only tasks matching the patterns in the weights dictionary will
    be considered, the patterns for which no tasks are found will be ignored.

    Args:
      weights: a dict assigning weights to task patterns, e.g.
        {'<task:success:.*': .1}.
    Returns:
      Str, a task string (without task tag).
    """
    # Organize matching tasks by pattern.
    matching_tasks = {}
    matching_weights = {}
    for pattern, weight in weights.items():
      for task, tasks in self.tasks_dict.items():
        if re.fullmatch(pattern, task, flags=self.RE_FLAGS):
          if pattern not in matching_tasks:
            matching_tasks[pattern] = []
            matching_weights[pattern] = weight
          matching_tasks[pattern].extend(tasks)

      return None, None
    pattern = random.choices(
        list(matching_weights.keys()), list(matching_weights.values()))[0]

    # Sample a task given a pattern.
    tasks = matching_tasks[pattern]
    if not tasks:
      raise ValueError('No tasks to sample of type %s in %s'
                       % (pattern, self.tasks_raw))
    task = random.choice(tasks)
    if not task:
      raise ValueError((
          'No tasks ("%s") is returned after choosing pattern %s and'
          ' returning random from "%s" from %s') % (
              task, pattern, matching_tasks[pattern], self.detailed_str()))
    return task, pattern

def fetch_question_answer(text):
  tasks = Tasks(text)
  results = []
  for i, (task_type, tasks) in enumerate(tasks.tasks_dict.items()):
    for task in tasks:
      t = Task(task)
      splits = t.get_splits('A:')
      for split in splits:
        question, answer = split
        question = question.strip()
        answer = answer.strip()
        results.append((i, task_type, question, answer))
  return results

def display_text(text):
  question_answers = fetch_question_answer(text)
  for i, task_type, question, answer in question_answers:
    print('Task %d (type: %s): %s %s' % (i, task_type, question, answer))

filepaths = tf.io.gfile.glob('/data/klx/hf_dataset/robovqa/gdm-robovqa/tfrecord/val/*')
dataset = tf.data.TFRecordDataset(filepaths)
np_iter = dataset.as_numpy_iterator()

raw_record = next(np_iter)
example = tf.train.SequenceExample()
example.ParseFromString(raw_record)

images = []
for bl in example.feature_lists.feature_list.get('images').feature:
  code = bl.bytes_list.value[0]
  image = tf.image.decode_jpeg(code).numpy()
  images.append(image)
print('Displayed items are:')
print('Video image frames (in GIF format)')
print('All VQA tasks, in format of "<Task num> (task type): <Question> <Answer>"')
display_image(images)
display_text(example.feature_lists.feature_list.get("texts").feature[0].bytes_list.value[0].decode('utf-8'))