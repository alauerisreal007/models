# # Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """Binary to run train and evaluation on object detection model."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from absl import flags

# import tensorflow.compat.v1 as tf
# from tensorflow.compat.v1 import estimator as tf_estimator

# from object_detection import model_lib

# flags.DEFINE_string(
#     'model_dir', None, 'Path to output model directory '
#     'where event and checkpoint files will be written.')
# flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
#                     'file.')
# flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
# flags.DEFINE_boolean('eval_training_data', False,
#                      'If training data should be evaluated for this job. Note '
#                      'that one call only use this in eval-only mode, and '
#                      '`checkpoint_dir` must be supplied.')
# flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
#                      'every n eval input examples, where n is provided.')
# flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
#                      'one of every n train input examples for evaluation, '
#                      'where n is provided. This is only used if '
#                      '`eval_training_data` is True.')
# flags.DEFINE_string(
#     'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
#     '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
#     'writing resulting metrics to `model_dir`.')
# flags.DEFINE_boolean(
#     'run_once', False, 'If running in eval-only mode, whether to run just '
#     'one round of eval vs running continuously (default).'
# )
# flags.DEFINE_integer(
#     'max_eval_retries', 0, 'If running continuous eval, the maximum number of '
#     'retries upon encountering tf.errors.InvalidArgumentError. If negative, '
#     'will always retry the evaluation.'
# )
# FLAGS = flags.FLAGS


# def main(unused_argv):
#   flags.mark_flag_as_required('model_dir')
#   flags.mark_flag_as_required('pipeline_config_path')
#   config = tf_estimator.RunConfig(model_dir=FLAGS.model_dir)

#   train_and_eval_dict = model_lib.create_estimator_and_inputs(
#       run_config=config,
#       pipeline_config_path=FLAGS.pipeline_config_path,
#       train_steps=FLAGS.num_train_steps,
#       sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
#       sample_1_of_n_eval_on_train_examples=(
#           FLAGS.sample_1_of_n_eval_on_train_examples))
#   estimator = train_and_eval_dict['estimator']
#   train_input_fn = train_and_eval_dict['train_input_fn']
#   eval_input_fns = train_and_eval_dict['eval_input_fns']
#   eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
#   predict_input_fn = train_and_eval_dict['predict_input_fn']
#   train_steps = train_and_eval_dict['train_steps']

#   if FLAGS.checkpoint_dir:
#     if FLAGS.eval_training_data:
#       name = 'training_data'
#       input_fn = eval_on_train_input_fn
#     else:
#       name = 'validation_data'
#       # The first eval input will be evaluated.
#       input_fn = eval_input_fns[0]
#     if FLAGS.run_once:
#       estimator.evaluate(input_fn,
#                          steps=None,
#                          checkpoint_path=tf.train.latest_checkpoint(
#                              FLAGS.checkpoint_dir))
#     else:
#       model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
#                                 train_steps, name, FLAGS.max_eval_retries)
#   else:
#     train_spec, eval_specs = model_lib.create_train_and_eval_specs(
#         train_input_fn,
#         eval_input_fns,
#         eval_on_train_input_fn,
#         predict_input_fn,
#         train_steps,
#         eval_on_train_data=False)

#     # Currently only a single Eval Spec is allowed.
#     tf_estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


# if __name__ == '__main__':
#   tf.app.run()

from absl import flags
import tensorflow as tf
import sys
from tensorflow.keras import layers, models

flags.DEFINE_string('model_dir', None, 'Path to output model directory where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False, 'If training data should be evaluated for this job.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of every n eval input examples.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample one of every n train input examples for evaluation.')
flags.DEFINE_string('checkpoint_dir', None, 'Path to directory holding a checkpoint.')
flags.DEFINE_boolean('run_once', False, 'If running in eval-only mode, whether to run just one round of eval vs running continuously.')
flags.DEFINE_integer('max_eval_retries', 0, 'If running continuous eval, the maximum number of retries upon encountering tf.errors.InvalidArgumentError.')
FLAGS = flags.FLAGS

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def train_model(model, train_dataset, epochs, steps_per_epoch):
    model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

def evaluate_model(model, eval_dataset, steps):
    return model.evaluate(eval_dataset, steps=steps)

def load_train_dataset(pipeline_config_path, sample_1_of_n_eval_on_train_examples):
    # Implement your data loading logic here
    return None  # Replace with actual dataset

def load_eval_dataset(pipeline_config_path, sample_1_of_n_eval_examples):
    # Implement your data loading logic here
    return None  # Replace with actual dataset

def main():
    # Parse the flags before accessing them
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    FLAGS(sys.argv)

    train_dataset = load_train_dataset(FLAGS.pipeline_config_path, FLAGS.sample_1_of_n_eval_on_train_examples)
    eval_dataset = load_eval_dataset(FLAGS.pipeline_config_path, FLAGS.sample_1_of_n_eval_examples)
    steps_per_epoch = FLAGS.num_train_steps

    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_model(model, train_dataset, epochs=10, steps_per_epoch=steps_per_epoch)

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            results = evaluate_model(model, train_dataset, steps=steps_per_epoch)
        else:
            results = evaluate_model(model, eval_dataset, steps=steps_per_epoch)
        print(f"Evaluation results: {results}")

if __name__ == '__main__':
    main()
