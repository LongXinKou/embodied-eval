
## 🔨 Setup

### gym
```
pip install gym==0.26.2
```

### ai2thor
```
pip instanll ai2thor==5.0.0
```
1. RuntimeError: vulkaninfo failed to run, please ask your administrator to install vulkaninfo (e.g. on Ubuntu systems this requires running sudo apt install vulkan-tools).
```python
'refer to "https://github.com/allenai/ai2thor/issues/1144"'
def unity_command(self, width, height, headless):
    fullscreen = 1 if self.fullscreen else 0

    command = self._build.executable_path

    if headless:
        command += " -batchmode -nographics"
    else:
        command += (
            " -screen-fullscreen %s -screen-quality %s -screen-width %s -screen-height %s"
            % (fullscreen, QUALITY_SETTINGS[self.quality], width, height)
        )

    if self.gpu_device is not None:
        # This parameter only applies to the CloudRendering platform.
        # Vulkan maps the passed in parameter to device-index - 1 when compared
        # to the nvidia-smi device ids
        device_index = self.gpu_device if self.gpu_device < 1 else self.gpu_device + 1
        command += " -force-device-index %d" % device_index

    return shlex.split(command)
```