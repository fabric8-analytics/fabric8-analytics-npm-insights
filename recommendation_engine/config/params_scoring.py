#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This file defines a parameter class that contains the model hyperparameters.

Copyright © 2018 Red Hat Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


class ScoringParams():
    """This class contains the model hyperparamters."""

    a = 1
    b = 0.01
    lambda_u = 0.1
    lambda_v = 10
    lambda_r = 1
    max_iter = 10
    M = 10
    m_num_factors = 50
    n_z = 50
