from gym.envs.registration import register

register(
    id = 'FlexPrinterEnv-v0',
    entry_point = 'environments.flex_outline:FlexPrinterEnv',
    max_episode_steps = 1000
)

register(
    id = 'FlexPrinterEnv-v1',
    entry_point = 'environments.flex_infill:FlexPrinterEnv',
    max_episode_steps = 1000
)