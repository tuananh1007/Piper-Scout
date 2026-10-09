from setuptools import find_packages, setup

package_name = "scout_piper_jepa"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/target_state.launch.py"]),
        ("share/" + package_name + "/config", ["config/target_memory.yaml", "config/e1_methods.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vu Tuan Anh",
    maintainer_email="tuananh05253@gmail.com",
    description="Piper-JEPA Stage A: dense target memory over V-JEPA features.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "target_state_node = scout_piper_jepa.target_state_node:main",
            "jepa_episode = scout_piper_jepa.episode:main",
            "jepa_annotate = scout_piper_jepa.annotate:main",
            "jepa_e1 = scout_piper_jepa.e1:main",
            "jepa_train = scout_piper_jepa.train:main",
            "jepa_latency = scout_piper_jepa.latency:main",
            "predictive_mpc_node = scout_piper_jepa.predictive_mpc_node:main",
            "jepa_ground = scout_piper_jepa.ground:main",
        ],
    },
)
