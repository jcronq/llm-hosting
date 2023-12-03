# llm-hosting
Experimentation with running LLMs on a local machine.

## Setup

You'll need to download a hugging face model into a directory `models/<model_name>`

There's a lot of defaults pointing to `models/mistral-7b-instruct` so I would recommend downloading this model.

## CLI
There's a cli (`pip install -e .` from this repo's root directory).

CLI is activated via `llm [action] [model directory]`

If you don't want to download a model before hand, you can also just use the hugging face model id in place of `model directory`.


## REST API
You can also run this repo as a rest api.  The rest api supports the OpenAI api.  It also introduces `/profiling/resources` endpoint 
which will return a json object containing a list of torch profiler traces.  You can load and view those traces by loading them into
`chrome://tracing` on the chrome browser
