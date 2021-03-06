#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""

import config.camvid_resnet50
import config.custom_xception
import config.custom_resnet50
import config.human_parsing_resnet50


CONFIG_MAP = {
    'custom_xception': config.custom_xception.CONFIG,
    'custom_resnet50': config.custom_resnet50.CONFIG,
    'camvid_resnet50': config.camvid_resnet50.CONFIG,
    'human_parsing_resnet50': config.human_parsing_resnet50.CONFIG
}
