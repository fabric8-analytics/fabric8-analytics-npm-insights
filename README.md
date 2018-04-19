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

## LICENSE

Licensed under the GNU GPL v3.0, copyright Red Hat Inc., 2018. Licenses for vendor code are included in the respective files/folders.
