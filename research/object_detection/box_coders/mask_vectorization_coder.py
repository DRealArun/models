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
NUM_CORNERS = 8


class MaskVectorizationCoder(box_coder.BoxCoder):
  """Mask vectorization coder."""

  def __init__(self, scale_factors=None):
    """Constructor for FasterRcnnBoxCoder.

    Args:
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
    """
    if scale_factors:
      assert len(scale_factors) == 10
      for scalar in scale_factors:
        assert scalar > 0
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 10

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
    o1 = boxes.get_field(fields.BoxListFields.o1) + EPSILON
    o2 = boxes.get_field(fields.BoxListFields.o2) + EPSILON
    o3 = boxes.get_field(fields.BoxListFields.o3) + EPSILON
    o4 = boxes.get_field(fields.BoxListFields.o4) + EPSILON
    o5 = boxes.get_field(fields.BoxListFields.o5) + EPSILON
    o6 = boxes.get_field(fields.BoxListFields.o6) + EPSILON
    o7 = boxes.get_field(fields.BoxListFields.o7) + EPSILON
    o8 = boxes.get_field(fields.BoxListFields.o8) + EPSILON

    ha += EPSILON
    wa += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha

    x1_a = (xcenter_a + (wa/2.0))
    y1_a = (ycenter_a + (ha/2.0))

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    # Calculate the offsets
    of1_a = self.find_distance(xcenter_a, ycenter_a, x1_a, y1_a) + EPSILON

    to1 = tf.log((o1)/ of1_a)
    to2 = tf.log((o2)/ of1_a)
    to3 = tf.log((o3)/ of1_a)
    to4 = tf.log((o4)/ of1_a)
    to5 = tf.log((o5)/ of1_a)
    to6 = tf.log((o6)/ of1_a)
    to7 = tf.log((o7)/ of1_a)
    to8 = tf.log((o8)/ of1_a)

    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      to1 *= self._scale_factors[2]
      to2 *= self._scale_factors[3]
      to3 *= self._scale_factors[4]
      to4 *= self._scale_factors[5]
      to5 *= self._scale_factors[6]
      to6 *= self._scale_factors[7]
      to7 *= self._scale_factors[8]
      to8 *= self._scale_factors[9]

    return tf.transpose(tf.stack([ty, tx, to1, to2, to3, to4, to5, to6, to7, to8]))

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ty, tx, to1, to2, to3, to4, to5, to6, to7, to8 = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      to1 /= self._scale_factors[2]
      to2 /= self._scale_factors[3]
      to3 /= self._scale_factors[4]
      to4 /= self._scale_factors[5]
      to5 /= self._scale_factors[6]
      to6 /= self._scale_factors[7]
      to7 /= self._scale_factors[8]
      to8 /= self._scale_factors[9]

    ha += EPSILON
    wa += EPSILON

    yc = ty * ha + ycenter_a
    xc = tx * wa + xcenter_a

    x1_a = (xcenter_a + (wa/2.0))
    y1_a = (ycenter_a + (ha/2.0))

    of1_a = self.find_distance(xcenter_a, ycenter_a, x1_a, y1_a) + EPSILON

    o1 = (tf.exp(to1) * of1_a) - EPSILON
    o2 = (tf.exp(to2) * of1_a) - EPSILON 
    o3 = (tf.exp(to3) * of1_a) - EPSILON
    o4 = (tf.exp(to4) * of1_a) - EPSILON
    o5 = (tf.exp(to5) * of1_a) - EPSILON
    o6 = (tf.exp(to6) * of1_a) - EPSILON
    o7 = (tf.exp(to7) * of1_a) - EPSILON
    o8 = (tf.exp(to8) * of1_a) - EPSILON
    
    cos_theta = math.cos(math.radians(45))
    sin_theta = math.sin(math.radians(45))
    pt1 = (xc+o1, yc)
    pt2 = (xc+(o2*cos_theta), yc-(o2*sin_theta))
    pt3 = (xc, yc-o3)
    pt4 = (xc-(o4*cos_theta), yc-(o4*sin_theta))
    pt5 = (xc-o5, yc)
    pt6 = (xc-(o6*cos_theta), yc+(o6*sin_theta))
    pt7 = (xc, yc+o7)
    pt8 = (xc+(o8*cos_theta), yc+(o8*sin_theta))
    
    return tf.transpose(tf.stack([pt1[1], pt1[0], pt2[1], pt2[0], pt3[1], pt3[0], pt4[1], pt4[0], pt5[1], pt5[0], pt6[1], pt6[0], pt7[1], pt7[0], pt8[1], pt8[0]]))
