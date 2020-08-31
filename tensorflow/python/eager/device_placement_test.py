# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for device placement."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util


class SoftDevicePlacementTest(test.TestCase):

  def setUp(self):
    context.context().soft_device_placement = True
    context.context().log_device_placement = True

  # TFDML #25509642
  @test_util.skip_dml
  @test_util.run_gpu_only
  def testDefaultPlacement(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = a + b
    with ops.device('CPU'):
      d = a + b
    self.assertIn(test_util.gpu_device_type(), c.device)
    self.assertIn('CPU', d.device)

  # TFDML #25509663
  @test_util.skip_dml
  @test_util.run_gpu_only
  def testUnsupportedDevice(self):
    gpu_type = test_util.gpu_device_type()
    a = constant_op.constant(1.0)
    b = constant_op.constant(2)
    s = constant_op.constant(list('hello world'))
    with ops.device('%s:0' % gpu_type):
      c = a + b
      t = s[a]
    self.assertIn('%s:0' % gpu_type, c.device)
    self.assertIn('CPU', t.device)

  # TFDML #25509659
  @test_util.skip_dml
  @test_util.run_gpu_only
  def testUnknownDevice(self):
    gpu_type = test_util.gpu_device_type()
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('%s:42' % gpu_type):
      c = a + b
    self.assertIn('%s:0' % gpu_type, c.device)

  def testNoGpu(self):
    if test_util.is_gpu_available():
      # CPU only test.
      return
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = a + b
    with ops.device('GPU'):
      d = a + b
    self.assertIn('CPU', c.device)
    self.assertIn('CPU', d.device)

  # TFDML #25509650
  @test_util.skip_dml
  @test_util.run_gpu_only
  def testNestedDeviceScope(self):
    gpu_type = test_util.gpu_device_type()
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('CPU:0'):
      with ops.device('GPU:42' % gpu_type):
        c = a + b
    # We don't support nested device placement right now.
    self.assertIn('%s:0' % gpu_type, c.device)


class ClusterPlacementTest(test.TestCase):

  def setUp(self):
    context.context().soft_device_placement = True
    context.context().log_device_placement = True
    workers, _ = test_util.create_local_cluster(2, 0)
    remote.connect_to_remote_host([workers[0].target, workers[1].target])

  def testNotFullySpecifiedTask(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('/job:worker'):
      c = a + b
    self.assertIn('/job:worker/replica:0/task:0', c.device)

  # TFDML #25576336
  @test_util.skip_dml
  def testRemoteUnknownDevice(self):
    gpu_type = test_util.gpu_device_type()
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    # Right now we don't support soft device place on remote worker.
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:worker/replica:0/task:0/device:%s:42' % gpu_type):
        c = a + b
        del c
      self.assertIn('unknown device', cm.exception.message)


if __name__ == '__main__':
  test.main()
