# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Center offset box coder.

Center offset box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  tof1, tof2, tof3 and tof4 check the calculation in encode function
  where x, y, of1, of2, of3, of4 denote the box's center coordinates, offsets of the corners
  respectively. Similarly, xa, ya, of1a, of2a, of3a, of4a denote the anchor's center
  coordinates, offsets of the corners. tx, ty, tof1, tof2, tof3, tof4 denote the anchor-encoded
  center and offsets respectively.

"""

import tensorflow as tf

from object_detection.core import box_coder
from object_detection.core import box_list
from object_detection.core import standard_fields as fields
import numpy as np
import math

EPSILON = 1e-8
NUM_CORNERS = 4


class CenterOffsetBoxCoder(box_coder.BoxCoder):
  """Center offset box coder."""

  def __init__(self, scale_factors=None):
    """Constructor for FasterRcnnBoxCoder.

    Args:
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
    """
    if scale_factors:
      assert len(scale_factors) == 7
      for scalar in scale_factors:
        assert scalar > 0
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 7

  def find_distance(self, x1, y1, x2, y2):
    return (((x1-x2)**2) + ((y1-y2)**2))**(0.5)

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    # ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    # Avoid NaN in division and log below.
    xcenter = boxes.get_field(fields.BoxListFields.center_x)
    ycenter = boxes.get_field(fields.BoxListFields.center_y)
    xmax = boxes.get_field(fields.BoxListFields.xmax)
    ymax = boxes.get_field(fields.BoxListFields.ymax)
    o1 = boxes.get_field(fields.BoxListFields.o1) + EPSILON
    o2 = boxes.get_field(fields.BoxListFields.o2) + EPSILON
    o3 = boxes.get_field(fields.BoxListFields.o3) + EPSILON

    ha += EPSILON
    wa += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha

    ymax_a = ycenter_a + (ha/2.0)
    xmax_a = xcenter_a + (wa/2.0)

    txmax = (xmax - xmax_a) / wa
    tymax = (ymax - ymax_a) / ha


    x1_a = (xcenter_a - (wa/2.0)) #min min
    y1_a = (ycenter_a - (ha/2.0))

    x2_a = (xcenter_a - (wa/2.0)) #min max
    y2_a = (ycenter_a + (ha/2.0))

    x3_a = (xcenter_a + (wa/2.0)) #max min
    y3_a = (ycenter_a - (ha/2.0))

    of1_a = self.find_distance(xcenter_a, ycenter_a, x1_a, y1_a) + EPSILON
    of2_a = self.find_distance(xcenter_a, ycenter_a, x2_a, y2_a) + EPSILON
    of3_a = self.find_distance(xcenter_a, ycenter_a, x3_a, y3_a) + EPSILON

    to1 = tf.log(o1 / of1_a)
    to2 = tf.log(o2 / of2_a)
    to3 = tf.log(o3 / of3_a)
    # Scales location targets as used in paper for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      tymax *= self._scale_factors[2]
      txmax *= self._scale_factors[3]
      to1 *= self._scale_factors[4]
      to2 *= self._scale_factors[4]
      to3 *= self._scale_factors[4]
    return tf.transpose(tf.stack([ty, tx, tymax, txmax, to1, to2, to3]))

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ty, tx, tymax, txmax, to1, to2, to3 = tf.unstack(tf.transpose(rel_codes))
    x1_a = (xcenter_a - (wa/2.0)) #min min
    y1_a = (ycenter_a - (ha/2.0))

    x2_a = (xcenter_a - (wa/2.0)) #min max
    y2_a = (ycenter_a + (ha/2.0))

    x3_a = (xcenter_a + (wa/2.0)) #max min
    y3_a = (ycenter_a - (ha/2.0))

    of1_a = self.find_distance(xcenter_a, ycenter_a, x1_a, y1_a) + EPSILON
    of2_a = self.find_distance(xcenter_a, ycenter_a, x2_a, y2_a) + EPSILON
    of3_a = self.find_distance(xcenter_a, ycenter_a, x3_a, y3_a) + EPSILON

    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      tymax /= self._scale_factors[2]
      txmax /= self._scale_factors[3]
      to1 /= self._scale_factors[4]
      to2 /= self._scale_factors[4]
      to3 /= self._scale_factors[4]
    o1 = tf.exp(to1) * of1_a
    o2 = tf.exp(to2) * of2_a
    o3 = tf.exp(to3) * of3_a

    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a

    ymax_a = ycenter_a + (ha/2.0)
    xmax_a = xcenter_a + (wa/2.0)

    ymax = tymax * ha + ymax_a
    xmax = txmax * wa + xmax_a

    angle = tf.atan(tf.abs(ymax-ycenter)/tf.abs(xmax-xcenter))*180/3.142
    cos_theta = tf.cos(angle)
    sin_theta = tf.sin(angle)
    pt1 = (xcenter-(o1*cos_theta), ycenter-(o1*sin_theta))
    pt2 = (xcenter-(o2*cos_theta), ycenter+(o2*sin_theta))
    pt3 = (xcenter+(o3*cos_theta), ycenter-(o3*sin_theta))
    pt4 = (xmax, ymax)
    return tf.transpose(tf.stack([pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1]]))
