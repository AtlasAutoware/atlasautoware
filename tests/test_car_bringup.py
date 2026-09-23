"""Construct bringup with an absent optional camera package, without ROS."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def test_optional_camera_package_is_not_resolved_during_construction(monkeypatch):
    class Action:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Description:
        def __init__(self):
            self.actions = []

        def add_action(self, action):
            self.actions.append(action)

    class FindPackageShare(Action):
        pass

    lookups = []

    def lookup(package):
        lookups.append(package)
        if package != 'f1tenth_gym_ros':
            raise LookupError(f'Optional package not installed: {package}')
        return '/workspace/share/f1tenth_gym_ros'

    exports = {
        'launch': {'LaunchDescription': Description},
        'launch.actions': dict.fromkeys(
            ['DeclareLaunchArgument', 'GroupAction', 'IncludeLaunchDescription'], Action),
        'launch.conditions': {'IfCondition': Action},
        'launch.launch_description_sources': {'PythonLaunchDescriptionSource': Action},
        'launch.substitutions': dict.fromkeys(
            ['LaunchConfiguration', 'PathJoinSubstitution', 'PythonExpression'], Action),
        'launch_ros': {},
        'launch_ros.actions': dict.fromkeys(['Node', 'SetRemap'], Action),
        'launch_ros.substitutions': {'FindPackageShare': FindPackageShare},
        'ament_index_python': {},
        'ament_index_python.packages': {'get_package_share_directory': lookup},
    }
    for name, attributes in exports.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[1] / 'launch' / 'car_bringup_launch.py'
    spec = importlib.util.spec_from_file_location('car_bringup_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    assert description.actions
    assert lookups == ['f1tenth_gym_ros']

    # Keep the optional driver in the launch graph, as a deferred substitution.
    def substitutions(value):
        if isinstance(value, FindPackageShare):
            yield value.args[0]
        elif isinstance(value, Action):
            yield from substitutions(value.args)
            yield from substitutions(value.kwargs)
        elif isinstance(value, dict):
            for child in value.values():
                yield from substitutions(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from substitutions(child)

    assert list(substitutions(description.actions)) == ['orbbec_camera']
