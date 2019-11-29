#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the constants for interaction with AWS.

Copyright © 2018 Red Hat Inc

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
import os

USE_CLOUD_SERVICES = os.environ.get("USE_CLOUD_SERVICES", True)
AWS_S3_ACCESS_KEY_ID = os.environ.get('AWS_S3_ACCESS_KEY_ID', '')
AWS_S3_SECRET_KEY_ID = os.environ.get('AWS_S3_SECRET_ACCESS_KEY', '')
S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME', 'cvae-insights')
