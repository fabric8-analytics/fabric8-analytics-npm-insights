[![Build Status](https://ci.centos.org/buildStatus/icon?job=devtools-fabric8-analytics-npm-insights-f8a-build-master)](https://ci.centos.org/job/devtools-fabric8-analytics-npm-insights-f8a-build-master/)

NPM companion package recommendations
-------------------------------------

This repository contains the code that is used to power NPM package companion
recommendations. The POC work around this lives in [this repo](https://github.com/fabric8-analytics/poc-npm-stack-analysis).
The approached used is based off of CVAE, see citation below.

```
Li, Xiaopeng, and James She. "Collaborative variational autoencoder for recommender systems."  
In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,  
pp. 305-314. ACM, 2017.
```

### Sample Request

```
Endpoint: /api/v1/companion_recommendation
Method: POST
Content-type: application/json
Body:
{
	"stack": ["express", "mongoose"]
}
```

### Sample Response

```
Content-type: application/json
{
    "missing_packages": [],
    "recommendations": [
        {
            "companion_probability": 0.9231991381908954,
            "package": "alaska",
            "tags": [
                "alaska",
                "koa",
                "mongoose",
                "react",
                "mvc",
                "web"
            ]
        },
        {
            "companion_probability": 0.8385970486757496,
            "package": "koa-grace-mongo",
            "tags": [
                "koa",
                "mongo",
                "grace-mongo"
            ]
        },
        {
            "companion_probability": 0.6536468661722873,
            "package": "ewares",
            "tags": [
                "express",
                "middleware",
                "express-middlewares"
            ]
        },
        {
            "companion_probability": 0.46540235321030543,
            "package": "peento",
            "tags": [
                "blog"
            ]
        },
        {
            "companion_probability": 0.3881866445153202,
            "package": "koa",
            "tags": [
                "web",
                "app",
                "http",
                "application",
                "framework",
                "middleware",
                "rack"
            ]
        },
        {
            "companion_probability": 0.37067553026014505,
            "package": "harvesterjs",
            "tags": [
                "json",
                "api",
                "jsonapi",
                "json-api",
                "framework",
                "rest",
                "restful"
            ]
        },
        {
            "companion_probability": 0.343074615791371,
            "package": "keystone",
            "tags": [
                "express",
                "web",
                "app",
                "cms",
                "admin",
                "framework",
                "mongoose",
                "gui",
                "site",
                "website",
                "forms"
            ]
        },
        {
            "companion_probability": 0.2857418338795869,
            "package": "strong-remoting",
            "tags": [
                "StrongLoop",
                "LoopBack",
                "Remoting",
                "REST"
            ]
        },
        {
            "companion_probability": 0.2750733954908234,
            "package": "mailgun-js",
            "tags": [
                "email",
                "mailgun"
            ]
        },
        {
            "companion_probability": 0.24752054711697755,
            "package": "requisition",
            "tags": [
                "request",
                "promise",
                "http",
                "https",
                "client"
            ]
        }
    ]
}
```

## Scripts to check if test code conformns to defined standards

### Code written in Python

#### Coding standards

- You can use scripts `check-PEP8-style.sh` and `check-python-docstyle.sh` to check if the code follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [PEP 257](https://www.python.org/dev/peps/pep-0257/) coding standards. These scripts can be run w/o any arguments:

```
./check-PEP8-style.sh
./check-python-docstyle.sh
```

The first script checks the indentation, line lengths, variable names, whitespace around operators etc. The second
script checks all documentation strings - its presense and format. Please fix any warnings and errors reported by these
scripts.

#### Code complexity measurement

The scripts `measure-cyclomatic-complexity.sh` and `measure-maintainability-index.sh` are used to measure code complexity. These scripts can be run w/o any arguments:

```
./measure-cyclomatic-complexity.sh
./measure-maintainability-index.sh
```

The first script measures cyclomatic complexity of all Python sources found in the repository. Please see [this table](https://radon.readthedocs.io/en/latest/commandline.html#the-cc-command) for further explanation how to comprehend the results.

The second script measures maintainability index of all Python sources found in the repository. Please see [the following link](https://radon.readthedocs.io/en/latest/commandline.html#the-mi-command) with explanation of this measurement.

#### Dead code detection

The script `detect-dead-code.sh` can be used to detect dead code in the repository. This script can be run w/o any arguments:

```
./detect-dead-code.sh
```

Please note that due to Python's dynamic nature, static code analyzers are likely to miss some dead code. Also, code that is only called implicitly may be reported as unused.

Because of this potential problems, only code detected with more than 90% of confidence is reported.

#### Common issues detection

The script `detect-common-errors.sh` can be used to detect common errors in the repository. This script can be run w/o any arguments:

```
./detect-common-errors.sh
```

Please note that only semantical problems are reported.

#### Check for scripts written in BASH

The script named `check-bashscripts.sh` can be used to check all BASH scripts (in fact: all files with the `.sh` extension) for various possible issues, incompatibilies, and caveats. This script can be run w/o any arguments:

```
./check-bashscripts.sh
```

Please see [the following link](https://github.com/koalaman/shellcheck) for further explanation, how the ShellCheck works and which issues can be detected.

## LICENSE

Licensed under the GNU GPL v3.0, copyright Red Hat Inc., 2018. Licenses for vendor code are included in the respective files/folders.
