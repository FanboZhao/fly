from typing import Sequence, Optional
import numpy as np
from dm_control import composer

from flybody.fruitfly import fruitfly

from direction_tracking_task import DirectionTracking

from flybody.tasks.arenas.hills import SineBumps, SineTrench
from flybody.tasks.pattern_generators import WingBeatPatternGenerator


def direction_tracking(wpg_pattern_path: str | None = None,
                        bumps_or_trench: str = 'bumps',
                        force_actuators: bool = False,
                        disable_legs: bool = True,
                        random_state: np.random.RandomState | None = None,
                        joint_filter: float = 0.,
                        **kwargs_arena):
    """Vision-guided flight tasks: 'bumps' and 'trench'.

    Args:
        wpg_pattern_path: Path to baseline wing beat pattern for WPG. If None,
            a simple approximate wing pattern will be generated in the
            WingBeatPatternGenerator class (could be used for testing).
        bumps_or_trench: Whether to create 'bumps' or 'trench' vision task.
        force_actuators: Whether to use force actuators or position actuators
            for everything other than wings. Wings always use force actuators.
        disable_legs: Whether to retract and disable legs. This includes
            removing leg DoFs, actuators, and sensors.
        random_state: Random state for reproducibility.
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        kwargs_arena: kwargs to be passed on to arena.

    Returns:
        Environment for vision-guided flight task.
    """

    if bumps_or_trench == 'bumps':
        arena = SineBumps
    elif bumps_or_trench == 'trench':
        arena = SineTrench
    else:
        raise ValueError("Only 'bumps' and 'trench' terrains are supported.")
    # Build fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = arena(**kwargs_arena)
    # Initialize a wing beat pattern generator.
    wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
    # Build task.
    time_limit = 0.4
    task = DirectionTracking(walker=walker,
                        arena=arena,
                        wbpg=wbpg,
                        time_limit=time_limit,
                        force_actuators=force_actuators,
                        disable_legs=disable_legs,
                        joint_filter=joint_filter,
                        floor_contacts=True,
                        floor_contacts_fatal=True,
                        )

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)