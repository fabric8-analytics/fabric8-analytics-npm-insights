"""Pypi bigquery implementation."""
import time
import os
from collections import Counter

from rudra.data_store.bigquery.base import BigqueryBuilder
from rudra.utils.pypi_parser import pip_req
from rudra.data_store.bigquery.base import DataProcessing
from rudra.utils.validation import BQValidation
from rudra import logger


class PyPiBigQuery(BigqueryBuilder):
    """PyPiBigQuery Implementation."""

    def __init__(self, *args, **kwargs):
        """Initialize PyPiBigQuery object."""
        super().__init__(*args, **kwargs)
        self.query_job_config.use_legacy_sql = False
        self.query_job_config.use_query_cache = True
        self.query_job_config.timeout_ms = 60000

        self.query = """
            SELECT con.content AS content
            FROM `bigquery-public-data.github_repos.contents` AS con
            INNER JOIN (SELECT files.id AS id
                        FROM `bigquery-public-data.github_repos.languages` AS langs
                        INNER JOIN `bigquery-public-data.github_repos.files` AS files
                        ON files.repo_name = langs.repo_name
                            WHERE REGEXP_CONTAINS(TO_JSON_STRING(language), r'(?i)python')
                            AND files.path LIKE '%requirements.txt'
                    ) AS L
            ON con.id = L.id;
        """


class PyPiBigQueryDataProcessing(DataProcessing):
    """Implementation data processing for pypi bigquery."""

    def __init__(self, big_query_instance=None, s3_client=None,
                 file_name='collated.json'):
        """Initialize the BigQueryDataProcessing object."""
        super().__init__(s3_client)
        self.big_query_instance = big_query_instance or PyPiBigQuery()
        self.big_query_content = list()
        self.counter = Counter()
        self.bucket_name = self.s3_client.bucket_name \
            if self.s3_client else'developer-analytics-audit-report'
        self.filename = '{}/big-query-data/{}'.format(
            os.getenv('DEPLOYMENT_PREFIX', 'dev'), file_name)

    def process(self, validate=False):
        """Process Pypi Bigquery response data."""
        bq_validation = BQValidation()
        logger.info("Running Bigquery for pypi synchronously")
        self.big_query_instance.run_query_sync()
        start_process_time = time.monotonic()
        for idx, obj in enumerate(self.big_query_instance.get_result()):
            start = time.monotonic()
            content = obj.get('content')
            packages = []
            if content:
                try:
                    packages = sorted(
                        {p for p in pip_req.parse_requirements(content)})
                    if validate:
                        packages = sorted(
                            bq_validation.validate_pypi(packages))
                except Exception as _exc:
                    logger.error("IGNORE: {}".format(_exc))
                    logger.error(
                        "Failed to parse content data {}".format(content))

                if packages:
                    pkg_string = ', '.join(packages)
                    logger.info("PACKAGES: {}".format(pkg_string))
                    self.counter.update([pkg_string])
                logger.info("Processed content in time: {} counter:{}".format(
                    (time.monotonic() - start), idx))
        logger.info("Processed All the manifests in time: {}".format(
            time.monotonic() - start_process_time))

        logger.info("updating file content")
        self.update_s3_bucket(data={'pypi': dict(self.counter.most_common())},
                              bucket_name=self.bucket_name,
                              filename=self.filename)

        logger.info("Succefully Processed the PyPiBigQuery")
