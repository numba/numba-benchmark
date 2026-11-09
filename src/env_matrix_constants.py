

ENVIRONMENTS = [
    {
        "os": "linux",
        "runner": "ubuntu-24.04",
        "python": "3.13",
        "numpy": "2.2",
        "spec": "numpy>=2.2,<2.3",
    },
    {
        "os": "linux",
        "runner": "ubuntu-24.04",
        "python": "3.12",
        "numpy": "2.1",
        "spec": "numpy>=2.1,<2.2",
    },
    {
        "os": "linux-arm",
        "runner": "ubuntu-24.04-arm",
        "python": "3.13",
        "numpy": "2.2",
        "spec": "numpy>=2.2,<2.3",
    },
    {
        "os": "linux-arm",
        "runner": "ubuntu-24.04-arm",
        "python": "3.12",
        "numpy": "2.1",
        "spec": "numpy>=2.1,<2.2",
    },
]

ENVIRONMENT_KEYS = [
    f"{environment['os']}|{environment['python']}|{environment['numpy']}"
    for environment in ENVIRONMENTS
]
