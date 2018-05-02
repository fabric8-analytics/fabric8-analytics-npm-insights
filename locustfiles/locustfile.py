#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the code to load test the NPM recommendation service.

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
import json
from locust import HttpLocust, TaskSet, events, task, web
from collections import Counter

stats = {"host-distribution": Counter()}


class StackAnalysisUserBehaviour(TaskSet):
    """This class defines the user behaviours."""

    def on_start(self):
        """Define the pre-commit hooks."""
        pass

    @task
    def trigger_stack_analysis_five_package_stack(self):
        """Simulate a stack analysis request."""
        stack = ["cli-color", "when", "moment",
                 "lodash", "optimist", "amqp", "async"]
        response = self.client.post("/", data=json.dumps({"stack": stack}),
                                    headers={'Content-type': 'application/json'})
        stats["host-distribution"][response.json()['HOSTNAME']] += 1
        print(stats['host-distribution'])


class StackAnalysisUserLocust(HttpLocust):
    """This class defines the params for the load testing piece."""

    task_set = StackAnalysisUserBehaviour
    min_wait = 10
    max_wait = 10
