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

"""Mask 8 point vectorization coder.

This encoder vectorizes the mask in a center-offset fashion. 
It follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  tof1 = log(of1 / anchor_diagonal)
  tof2 = log(of2 / anchor_diagonal)
  tof3 = log(of3 / anchor_diagonal)
  tof4 = log(of4 / anchor_diagonal)
  where x, y, w, h, of1, of2, of3, of4 denote the box's center coordinates, width, height and 
  offsets from the center respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  tx, ty, tw, th, tof1, tof2, tof3, tof4 denote the anchor-encoded center, width, height 
  and offsets respectively.

"""

import tensorflow as tf

from object_detection.core import box_coder
from object_detection.core import box_list
from object_detection.core import standard_fields as fields
import numpy as np
import math

EPSILON = 1e-8
NUM_CORNERS = 8


class Mask8PointVectorizationCoder(box_coder.BoxCoder):
  """Mask 8 point vectorization coder."""

  def __init__(self, scale_factors=None):
    """Constructor for Mask8PointVectorizationCoder.

    Args:
      scale_factors: List of 8 positive scalars to scale ty, tx, th, tw, tof1,
      tof2, tof3 and tof4.
        If set to None, does not perform scaling.
    """
    if scale_factors:
      assert len(scale_factors) == 8
      for scalar in scale_factors:
        assert scalar > 0
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 8

  def _find_diagonal(self, width, height):
    return (((width)**2) + ((height)**2))**(0.5)

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
    def _get_perpendicular_distances(pt1s, ms, pt2):
      """Finds the perpendicular distance between a point and a line.
      Args:
        pt1s: points corresponding to multiple lines
        ms  : slope of the different lines
        pt2: point from which perpendicular distances to lines are to be found.
      Returns:
        offset: perpendicular distances between lines and the point.
      """
      pt1_xs, pt1_ys = pt1s[:,0], pt1s[:,1] #line
      pt2_x, pt2_y = pt2[0], pt2[1] #point
      cs = tf.subtract(pt1_ys, tf.multiply(ms,pt1_xs))
      # Ax+By+C=0 -> y=mx+c -> y-mx-c=0 -> -mx+y-c=0 -> A = -m, B=1, C=-c
      # perpendicular distance = |A*x2 + B*y2 + C| /sqrt((A**2+B**2))
      As = -ms
      multiplier = tf.shape(ms)
      Bs = tf.reshape(tf.tile([1.0], [multiplier[0]]), [multiplier[0], ])
      Cs = -cs
      offsets = tf.abs((As*pt2_x + Bs*pt2_y + Cs))/((As**2+Bs**2)**(0.5))
      return offsets

    def _get_8_point_mask(mask_tensor):
      """Finds the safe octagonal encoding of the polygon.
      Args:
        polygon: list of points representing the mask
        h      : height of the image
        w      : width of the image
      Returns:
        mask vector: [cx, cy, w, h, of1, of2, of3, of4]
      """
  #     mask_tensor = tf.cast(mask_tensor, tf.int64)
      num_elems = tf.shape(mask_tensor)
      h = tf.cast(num_elems[0], tf.float32)
      w = tf.cast(num_elems[1], tf.float32)
      # find all the pixel locations which are not zero.
      zero_tensor = tf.constant([0], tf.float32)
      max_bound_width = w
      max_bound_height = h
      non_zero_indices = tf.where(tf.not_equal(mask_tensor, zero_tensor))
      rss = tf.cast(non_zero_indices[:,0], tf.float32) #y_values
      css = tf.cast(non_zero_indices[:,1], tf.float32) #x_values
      
      # correct the out of bound values
      rss = tf.where(tf.less(rss, zero_tensor), tf.zeros_like(rss), rss)
      css = tf.where(tf.less(css, zero_tensor), tf.zeros_like(css), css)
      multiplier = tf.shape(rss)
      rss = tf.where(tf.greater(rss, max_bound_height), tf.reshape(tf.tile([h], [multiplier[0]]), [multiplier[0], ]), rss)
      css = tf.where(tf.greater(css, max_bound_width), tf.reshape(tf.tile([w], [multiplier[0]]), [multiplier[0], ]), css)
      
      # Add the two tensors and subtract the two tensors.
      sum_values = tf.add(css, rss)
      diff_values = tf.subtract(css, rss)
      # diff_values = tf.Print(diff_values, [diff_values], "Difference values")
      # sum_values = tf.Print(sum_values, [sum_values], "Sum values")
      # css = tf.Print(css, [css], "CSS")
      # rss = tf.Print(rss, [rss], "RSS")
      
      # Find the extreme values
      xmin = tf.reduce_min(css)
      xmax = tf.reduce_max(css)
      ymin = tf.reduce_min(rss)
      ymax = tf.reduce_max(rss)
      width  = xmax - xmin
      height = ymax - ymin
      center_x  = xmin + (width/2) 
      center_y  = ymin + (height/2)
      center = [center_x, center_y]
      
      min_sum_indices = tf.where(tf.equal(sum_values, tf.reduce_min(sum_values)))
      pt_p_min = [tf.gather(css,min_sum_indices)[0][0], tf.gather(rss,min_sum_indices)[0][0]]
      
      max_sum_indices = tf.where(tf.equal(sum_values, tf.reduce_max(sum_values)))
      pt_p_max = [tf.gather(css,max_sum_indices)[0][0], tf.gather(rss,max_sum_indices)[0][0]]
      
      min_diff_indices = tf.where(tf.equal(diff_values, tf.reduce_min(diff_values)))
      pt_n_min = [tf.gather(css,min_diff_indices)[0][0], tf.gather(rss,min_diff_indices)[0][0]]
      
      max_diff_indices = tf.where(tf.equal(diff_values, tf.reduce_max(diff_values)))
      pt_n_max = [tf.gather(css,max_diff_indices)[0][0], tf.gather(rss,max_diff_indices)[0][0]]
      
      pts = tf.convert_to_tensor([pt_p_min, pt_n_min, pt_p_max, pt_n_max])
      ms = tf.convert_to_tensor([-1., +1., -1.,  +1.]) #Slope of the tangents
      offsets = _get_perpendicular_distances(pts, ms, center)
      mask_vector = tf.stack([center_x/w, center_y/h, width/w, height/h, offsets[0]/w, offsets[1]/w, offsets[2]/w, offsets[3]/w])
      mask_vector = tf.reshape(mask_vector, (1, tf.shape(mask_vector)[0]))
      # mask_vector = tf.Print(mask_vector, [mask_vector], "MASK VECTOR", summarize=mask_vector.shape[1])
      return mask_vector

    def process_masks(mask_db):
      mask_list = tf.convert_to_tensor(mask_db)
      def condition(i, vectorized_masks):
          shape_temp = tf.shape(mask_list)
          # shape_temp = tf.Print(shape_temp, [shape_temp], "SHAPEEEEEEEEEEEEE")
          return tf.less(i, shape_temp[0])
                
      def body(i, vectorized_masks):
          current_mask = mask_list[i,:,:]
          def non_zero_mask_vectorization():
              mask_vec = _get_8_point_mask(current_mask)
              return mask_vec
          def zero_mask_vectorization():
              mask_vec = tf.zeros([1,8], dtype=tf.float32)
              return mask_vec
          # If the mask is empty then return and empty mask vector
          vectorized_mask = tf.cond(tf.equal(tf.reduce_sum(current_mask), 0), zero_mask_vectorization, non_zero_mask_vectorization)
          shape_vectorized_mask = tf.shape(vectorized_mask)[0]
          assert_op = tf.Assert(tf.equal(shape_vectorized_mask, 1), [vectorized_mask])
          with tf.control_dependencies([assert_op]):
              vectorized_masks = tf.concat([vectorized_masks, vectorized_mask], 0)
          return [tf.add(i, 1), vectorized_masks]
      
      i = tf.constant(0, dtype=tf.int32)
      vectorized_masks = tf.zeros([0,8], dtype=tf.float32)
      i, vectorized_masks = tf.while_loop(condition, body, [i, vectorized_masks], shape_invariants=[i.get_shape(), tf.TensorShape([None, 8])])
      return vectorized_masks
      
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    # ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    # xcenter = boxes.get_field(fields.BoxListFields.center_x)
    masks = boxes.get_field(fields.BoxListFields.masks)
    vectorized_masks = process_masks(masks)
    xcenter, ycenter, w, h, of1, of2, of3, of4 = tf.unstack(vectorized_masks, axis=1)

    # ycenter = boxes.get_field(fields.BoxListFields.center_y)
    # h = boxes.get_field(fields.BoxListFields.height)
    # w = boxes.get_field(fields.BoxListFields.width)
    # Avoid NaN in division and log below.

    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)

    # of1 = boxes.get_field(fields.BoxListFields.of1) 
    # of2 = boxes.get_field(fields.BoxListFields.of2)
    # of3 = boxes.get_field(fields.BoxListFields.of3)
    # of4 = boxes.get_field(fields.BoxListFields.of4)

    # Calculate the offsets
    anchor_diagonal = self._find_diagonal(wa, ha) + EPSILON

    tof1 = tf.log((of1 + EPSILON)/ anchor_diagonal)
    tof2 = tf.log((of2 + EPSILON)/ anchor_diagonal)
    tof3 = tf.log((of3 + EPSILON)/ anchor_diagonal)
    tof4 = tf.log((of4 + EPSILON)/ anchor_diagonal)

    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
      tof1 *= self._scale_factors[4]
      tof2 *= self._scale_factors[5]
      tof3 *= self._scale_factors[6]
      tof4 *= self._scale_factors[7]

    return tf.transpose(tf.stack([ty, tx, th, tw, tof1, tof2, tof3, tof4]))

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    def _get_intersecting_points(vert_hors, eq1s, pts, ms):
        """Finds the points of intersections of a lines with vertical or 
           horizontal lines.
        Args:
          vert_hors: "vert" or "hor" represents intersection is to be found 
                    with respect to which line.
          eq1s: equations of vertical or horizontal lines.
          pts : points on different lines.
          ms : slopes of lines.
        Returns:
          point of intersection of line1 with vertical or horizontal.
        """
        pt_xs, pt_ys = tf.transpose(pts[:,0,:]), tf.transpose(pts[:,1,:]) #(24,)
        pt_xs = tf.reshape(pt_xs, [-1])
        pt_ys = tf.reshape(pt_ys, [-1])
        ms = tf.reshape(tf.transpose(ms),(tf.shape(pt_xs)[0],))
        eq1s = tf.reshape(tf.transpose(eq1s), (tf.shape(pt_xs)[0],))
        cs = tf.subtract(pt_ys, tf.multiply(ms, pt_xs))

        x_cor_possible_vals = tf.truediv(tf.subtract(eq1s,cs),ms)
        x_cor = tf.where(tf.equal(vert_hors, "vert"), eq1s, x_cor_possible_vals)

        y_cor_possible_vals = tf.add(tf.multiply(ms, eq1s), cs)
        y_cor = tf.where(tf.equal(vert_hors, "vert"), y_cor_possible_vals, eq1s)
        return tf.reshape(tf.stack([x_cor, y_cor], axis=1), (-1,8,2))
        
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    # rel_codes = tf.Print(rel_codes, [rel_codes], "Codes:", summarize=10)
    ty, tx, th, tw, tof1, tof2, tof3, tof4 = tf.unstack(tf.transpose(rel_codes))
    shape_int = tf.shape(ty)
    num_masks = shape_int[0]
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      th /= self._scale_factors[2]
      tw /= self._scale_factors[3]
      tof1 /= self._scale_factors[4]
      tof2 /= self._scale_factors[5]
      tof3 /= self._scale_factors[6]
      tof4 /= self._scale_factors[7]

    ha += EPSILON
    wa += EPSILON
    width = tf.exp(tw) * wa
    height = tf.exp(th) * ha
    center_y = ty * ha + ycenter_a
    center_x = tx * wa + xcenter_a

    anchor_diagonal = self._find_diagonal(wa, ha) + EPSILON

    of1 = (tf.exp(tof1) * anchor_diagonal) - EPSILON
    of2 = (tf.exp(tof2) * anchor_diagonal) - EPSILON 
    of3 = (tf.exp(tof3) * anchor_diagonal) - EPSILON
    of4 = (tf.exp(tof4) * anchor_diagonal) - EPSILON

    cos = math.cos(math.radians(45))
    sin = math.cos(math.radians(45))

    pts =  [[]]*4 # (4,2,3)
    pts[0] = [center_x-of1*cos, center_y-of1*sin] #(2,3)
    pts[1] = [center_x-of2*cos, center_y+of2*sin]
    pts[2] = [center_x+of3*cos, center_y+of3*sin]
    pts[3] = [center_x+of4*cos, center_y-of4*sin]
    xmin = center_x - (0.5*width) # (3,)
    xmax = center_x + (0.5*width)
    ymin = center_y - (0.5*height)
    ymax = center_y + (0.5*height)
    points = tf.convert_to_tensor([pts[0], pts[1], pts[1], pts[2], pts[2], pts[3], pts[3], pts[0]]) # (8,2,3)
    eq1s = tf.convert_to_tensor([xmin, xmin, ymax, ymax, xmax, xmax, ymin, ymin]) # (8,3)
    vert_or_hors = tf.tile(["vert", "vert", "hor", "hor", "vert", "vert", "hor", "hor"], [num_masks]) #(24,)
    m1 = (pts[2][1][:]-pts[0][1][:])/(pts[2][0][:]-pts[0][0][:]) #(3,)
    m2 = (pts[3][1][:]-pts[1][1][:])/(pts[3][0][:]-pts[1][0][:]) #(3,)
    ms = [-1/m1, -1/m2, -1/m2, -1/m1, -1/m1, -1/m2, -1/m2, -1/m1] #(8,3)
    intersecting_pts = _get_intersecting_points(vert_or_hors, eq1s, points, ms) # (N,8,2)
    pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8 = tf.unstack(intersecting_pts, axis=1)
    return tf.transpose(tf.stack([pt1[:,1], pt1[:,0], pt2[:,1], pt2[:,0], pt3[:,1], pt3[:,0], pt4[:,1], pt4[:,0], pt5[:,1], pt5[:,0], pt6[:,1], pt6[:,0], pt7[:,1], pt7[:,0], pt8[:,1], pt8[:,0]]))
