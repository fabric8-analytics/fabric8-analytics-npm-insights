"""Check and promote the new model based on output parameters."""
import os
import json
import logging
import daiquiri
import ruamel.yaml
from github import Github

MODEL_VERSION = os.environ.get("MODEL_VERSION", "")
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
DEPLOYMENT_PREFIX = os.environ.get('DEPLOYMENT_PREFIX', 'dev')

UPSTREAM_REPO_NAME = 'openshiftio'
FORK_REPO_NAME = 'developer-analytics-bot'
PROJECT_NAME = 'saas-analytics'
YAML_FILE_PATH = 'bay-services/f8a-npm-insights.yaml'

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


class ModelPromotion:
    """Promte the new model after comparing it with existing one."""

    def __init__(self, s3_client):
        """Set s3 client and promotion creteria."""
        self.s3_client = s3_client
        self.promotion_creteria = ''

    def promote(self, hyper_params):
        """Check and promote model."""
        deployment_prefix_to_type_map = {
            'STAGE': 'staging',
            'PROD': 'production'
        }

        deployment_type = deployment_prefix_to_type_map.get(DEPLOYMENT_PREFIX.upper(), None)
        assert deployment_type is not None, f'Invalid DEPLOYMENT_PREFIX: {DEPLOYMENT_PREFIX}'

        upstream_repo = Github(GITHUB_TOKEN).get_repo(f'{UPSTREAM_REPO_NAME}/{PROJECT_NAME}')
        deployed_data, yaml_data, latest_commit_hash = self._read_deployed_data(
            upstream_repo, deployment_type)
        if self._is_better_model(deployed_data['hyperparams'], hyper_params):
            params = hyper_params.copy()
            params.update({'promotion_criteria': str(self.promotion_creteria)})
            branch_name, commit_message = self._create_branch_and_update_yaml(
                deployment_type, deployed_data, yaml_data, params, latest_commit_hash)
            hyper_params_formated = self._build_hyper_params_message(hyper_params)
            prev_hyper_params_formated = self._build_hyper_params_message(
                deployed_data['hyperparams'])
            body = f'''Current deployed model details:
- Model version :: `{deployed_data['version']}`
{prev_hyper_params_formated}

New model details:
- Model version :: `{MODEL_VERSION}`
{hyper_params_formated}

Criteria for promotion is `{self.promotion_creteria}`
'''
            pr = upstream_repo.create_pull(title=commit_message, body=self._format_body(body),
                                           head=f'{FORK_REPO_NAME}:{branch_name}',
                                           base='refs/heads/master')
            logger.info('Raised SAAS %s for review', pr)
        else:
            logger.warn('Ignoring latest model %s is not promoted', MODEL_VERSION)

    def _is_better_model(self, current_params, new_params):
        """Decides whether model with given params is better than current deployed model."""
        logger.info('Deployed model %s and new model is %s', current_params, new_params)
        self.promotion_creteria = 'Always move to latest model'
        return True

    def _read_deployed_data(self, upstream_repo, deployment_type):
        """Read deployed data like yaml file, hyper params, model version."""
        upstream_latest_commit_hash = upstream_repo.get_commits()[0].sha
        logger.info('Upstream latest commit hash: %s', upstream_latest_commit_hash)

        contents = upstream_repo.get_contents(YAML_FILE_PATH, ref=upstream_latest_commit_hash)
        yaml_dict = ruamel.yaml.load(
            contents.decoded_content.decode('utf8'), ruamel.yaml.RoundTripLoader)

        deployed_version = self._get_deployed_model_version(yaml_dict, deployment_type)
        deployed_file_path = f'{deployed_version}/intermediate-model/hyperparameters.json'
        deployed_hyperparams = self.s3_client.read_json_file(deployed_file_path)

        deployed_data = {
            'version': deployed_version,
            'hyperparams': deployed_hyperparams
        }
        yaml_data = {
            'content_sha': contents.sha,
            'dict': yaml_dict
        }

        return deployed_data, yaml_data, upstream_latest_commit_hash

    def _update_yaml_data(self, yaml_dict, deployment_type, model_version, hyper_params):
        """Update the yaml file for given deployment with model data and description as comments."""
        environments = yaml_dict.get('services', [{}])[0].get('environments', [])
        hyper_params = {k: str(v) for k, v in hyper_params.items()}
        for index, env in enumerate(environments):
            if env.get('name', '') == deployment_type:
                yaml_dict['services'][0]['environments'][index]['comments'] = hyper_params
                yaml_dict['services'][0]['environments'][index]['parameters']['MODEL_VERSION'] = \
                    model_version
                break

        return ruamel.yaml.dump(yaml_dict, Dumper=ruamel.yaml.RoundTripDumper)

    def _get_deployed_model_version(self, yaml_dict, deployment_type):
        """Read deployment yaml and return the deployed model verison."""
        model_version = None
        environments = yaml_dict.get('services', [{}])[0].get('environments', [])
        for env in environments:
            if env.get('name', '') == deployment_type:
                model_version = env.get('parameters', {}).get('MODEL_VERSION', '')
                break

        if model_version is None:
            raise Exception(f'Model version could not be found for deployment {deployment_type}')

        logger.info('Model version: %s for deployment: %s', model_version, deployment_type)
        return model_version

    def _build_hyper_params_message(self, hyper_params):
        """Build hyper params data string used for PR description and in yaml comments."""
        return '- Hyper parameters :: {}'.format(json.dumps(hyper_params, indent=4, sort_keys=True))

    def _format_body(self, body):
        """Format PR body string to replace decorators."""
        return body.replace('"', '').replace('{', '').replace('}', '').replace(',', '')

    def _create_branch_and_update_yaml(self, deployment_type, deployed_data, yaml_data,
                                       hyper_params, latest_commit_hash):
        """Create branch and update yaml content on fork repo."""
        # Update yaml model version for the given deployment
        new_yaml_data = self._update_yaml_data(
            yaml_data['dict'], deployment_type, MODEL_VERSION, hyper_params)
        logger.info('Modified yaml data, new length: %d', len(new_yaml_data))

        # Connect to fabric8 analytic repo & get latest commit hash
        f8a_repo = Github(GITHUB_TOKEN).get_repo(f'{FORK_REPO_NAME}/{PROJECT_NAME}')
        logger.info('f8a fork repo: %s', f8a_repo)

        # Create a new branch on f8a repo
        branch_name = f'bump_f8a-pypi-insights_for_{deployment_type}_to_{MODEL_VERSION}'
        branch = f8a_repo.create_git_ref(f'refs/heads/{branch_name}', latest_commit_hash)
        logger.info('Created new branch [%s] at [%s]', branch, latest_commit_hash)

        # Update the yaml content in branch on f8a repo
        commit_message = f'Bump up f8a-pypi-insights for {deployment_type} from ' \
                         f'{deployed_data["version"]} to {MODEL_VERSION}'
        update = f8a_repo.update_file(
            YAML_FILE_PATH, commit_message, new_yaml_data, yaml_data['content_sha'],
            branch=f'refs/heads/{branch_name}')
        logger.info('New yaml content hash %s', update['commit'].sha)

        return branch_name, commit_message
