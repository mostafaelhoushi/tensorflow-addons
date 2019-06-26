# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np
from tensorflow_addons.utils.resource_loader import get_path_to_datafile

_gelu_op_so = tf.load_op_library(
    get_path_to_datafile("custom_ops/gelu/_gelu_ops.so"))
gelu = _gelu_op_so.gelu

# reference function to compare against
# source: https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L264
def gelu_ref(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

class GeluTest(tf.test.TestCase):
  def testGelu(self):
    # create array with random dimensions and random values
    N, H, W, C = np.random.randint(1, 128), np.random.randint(1, 224), np.random.randint(1, 224), np.random.randint(1, 3)
    x = tf.constant(np.random.randn(N, H, W, C), dtype=tf.float32)

    with self.test_session():
        # run both versions of gelu
        y = gelu(x) # bert_ops.gelu(..) is implemented as a custom op in C++
        y_ref = gelu_ref(x)

        self.assertAllClose(y, y_ref)

  def testGeluGrad(self):
    x = tf.ones((2, 2))

    # create array with random dimensions and random values
    N, H, W, C = np.random.randint(1, 128), np.random.randint(1, 224), np.random.randint(1, 224), np.random.randint(1, 3)
    dL_dy = tf.constant(np.random.randn(N, H, W, C), dtype=tf.float32)
    x = tf.constant(np.random.randn(N, H, W, C), dtype=tf.float32)

    with self.test_session(): #with tf.Session() as sess:
        dy_dx = None
        dy_ref_dx = None
        with tf.GradientTape() as t:
            t.watch(x)
            y = gelu(x)

            # Derivative of y with respect to the original input tensor x
            dy_dx = t.gradient(y, x)

        with tf.GradientTape() as t:
            t.watch(x)
            y_ref = gelu_ref(x)

            # Derivative of y with respect to the original input tensor x
            dy_ref_dx = t.gradient(y_ref, x)
        
        self.assertAllClose(dy_dx, dy_ref_dx, atol=1e-02)


if __name__ == "__main__":
  tf.test.main()
