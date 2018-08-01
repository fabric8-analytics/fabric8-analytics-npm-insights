#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program is the abstract class for the online recommender logic.

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
from abc import abstractmethod, ABCMeta


class AbstractRecommender:
    """This defines the interface for the online recommender."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, _user_stack, _companion_threshold):
        """Generate companion recommendation for this user's stack."""
        raise NotImplementedError()
