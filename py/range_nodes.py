#Range Nodes from abandoned repo https://github.com/M1kep/Comfy_KepListStuff


from _decimal import Context, getcontext
from decimal import Decimal
from typing import Iterator, List, Tuple, Dict, Any, Union, Optional

import numpy as np

from .status_utils import push_node_status
from .utils import (
    error_if_mismatched_list_args,
    zip_with_fill,
)


def _first_or_none(value):
    """Hidden inputs on INPUT_IS_LIST nodes arrive as lists — unwrap to scalar."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value

custom_context = Context(prec=8)


class Soze_IntRangeNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "stop": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "step": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "range"

    def build_range(
        self, start: List[int], stop: List[int], step: List[int], end_mode: List[str], unique_id=None,
    ) -> Tuple[List[int], List[int]]:
        unique_id = _first_or_none(unique_id)
        # Drop unique_id from locals() before validation — it's not a list-arg.
        validate_args = {k: v for k, v in locals().items() if k not in ("unique_id", "self", "__class__", "validate_args")}
        error_if_mismatched_list_args(validate_args)

        ranges = []
        range_sizes = []
        for e_start, e_stop, e_step, e_end_mode in zip_with_fill(
            start, stop, step, end_mode
        ):
            if e_end_mode == "Inclusive":
                e_stop += 1
            vals = list(range(e_start, e_stop, e_step))
            ranges.extend(vals)
            range_sizes.append(len(vals))

        push_node_status(unique_id, f"OK: {len(ranges)} value(s) across {len(range_sizes)} range(s); sizes={range_sizes}")
        return ranges, range_sizes


class Soze_IntNumStepsRangeNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "stop": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "num_steps": (
                    "INT",
                    {"default": 0, "min": -4096, "max": 4096, "step": 1},
                ),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
                "allow_uneven_steps": (["True", "False"], {"default": "False"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "range"

    def build_range(
        self,
        start: List[int],
        stop: List[int],
        num_steps: List[int],
        end_mode: List[str],
        allow_uneven_steps: List[str],
        unique_id=None,
    ) -> Tuple[List[int], List[int]]:
        unique_id = _first_or_none(unique_id)
        if len(allow_uneven_steps) > 1:
            raise Exception("List input for allow_uneven_steps is not supported.")

        validate_args = {k: v for k, v in locals().items() if k not in ("unique_id", "self", "__class__", "validate_args")}
        error_if_mismatched_list_args(validate_args)

        ranges = []
        range_sizes = []
        for e_start, e_stop, e_num_steps, e_end_mode in zip_with_fill(
            start, stop, num_steps, end_mode
        ):
            direction = 1 if e_stop > e_start else -1
            if e_end_mode == "Exclusive":
                e_stop -= direction

            # Check for uneven steps
            step_size = (e_stop - e_start) / (e_num_steps - 1)
            if not allow_uneven_steps[0] == "True" and step_size != int(step_size):
                raise ValueError(
                    f"Uneven steps detected for start={e_start}, stop={e_stop}, num_steps={e_num_steps}."
                )

            vals = (
                np.rint(np.linspace(e_start, e_stop, e_num_steps)).astype(int).tolist()
            )
            ranges.extend(vals)
            range_sizes.append(len(vals))

        push_node_status(unique_id, f"OK: {len(ranges)} value(s) across {len(range_sizes)} range(s); sizes={range_sizes}")
        return ranges, range_sizes


class Soze_FloatRangeNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "start": ( "FLOAT", {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 0.01},),
                "stop": ("FLOAT", {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 0.01}),
                "step": ("FLOAT", {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 0.01}),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "range"

    @staticmethod
    def _decimal_range(
        start: Decimal, stop: Decimal, step: Decimal, inclusive: bool
    ) -> Iterator[float]:
        ret_val = start
        if inclusive:
            stop = stop + step

        direction = 1 if step > 0 else -1
        # while ret_val < stop:
        #     yield float(ret_val)
        #     ret_val += step
        while (ret_val - stop) * direction < 0:
            yield float(ret_val)
            ret_val += step

    def build_range(
        self,
        start: List[Union[float, Decimal]],
        stop: List[Union[float, Decimal]],
        step: List[Union[float, Decimal]],
        end_mode: List[str],
        unique_id=None,
    ) -> Tuple[List[float], List[int]]:
        unique_id = _first_or_none(unique_id)
        validate_args = {k: v for k, v in locals().items() if k not in ("unique_id", "self", "__class__", "validate_args")}
        error_if_mismatched_list_args(validate_args)
        getcontext().prec = 12

        start = [Decimal(s) for s in start]
        stop = [Decimal(s) for s in stop]
        step = [Decimal(s) for s in step]

        ranges = []
        range_sizes = []
        for e_start, e_stop, e_step, e_end_mode in zip_with_fill(
            start, stop, step, end_mode
        ):
            vals = list(
                self._decimal_range(e_start, e_stop, e_step, e_end_mode == "Inclusive")
            )
            ranges.extend(vals)
            range_sizes.append(len(vals))

        push_node_status(unique_id, f"OK: {len(ranges)} value(s) across {len(range_sizes)} range(s); sizes={range_sizes}")
        return ranges, range_sizes


class Soze_FloatNumStepsRangeNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "start": ( "FLOAT", {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 0.01},),
                "stop": ("FLOAT", {"default": 0.0, "min": -4096.0, "max": 4096.0, "step": 0.01}),
                "num_steps": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "range"

    @staticmethod
    def _decimal_range(
        start: Decimal, stop: Decimal, num_steps: int
    ) -> Iterator[float]:
        step = (stop - start) / (num_steps - 1)
        direction = 1 if step > 0 else -1

        ret_val = start
        for _ in range(num_steps):
            if (
                ret_val - stop
            ) * direction > 0:  # Ensure we don't exceed the 'stop' value
                break
            yield float(ret_val)
            ret_val += step

    def build_range(
        self,
        start: List[Union[float, Decimal]],
        stop: List[Union[float, Decimal]],
        num_steps: List[int],
        unique_id=None,
    ) -> Tuple[List[float], List[int]]:
        unique_id = _first_or_none(unique_id)
        validate_args = {k: v for k, v in locals().items() if k not in ("unique_id", "self", "__class__", "validate_args")}
        error_if_mismatched_list_args(validate_args)
        getcontext().prec = 12

        start = [Decimal(s) for s in start]
        stop = [Decimal(s) for s in stop]

        ranges = []
        range_sizes = []
        for e_start, e_stop, e_num_steps in zip_with_fill(start, stop, num_steps):
            vals = list(self._decimal_range(e_start, e_stop, e_num_steps))
            ranges.extend(vals)
            range_sizes.append(len(vals))

        push_node_status(unique_id, f"OK: {len(ranges)} value(s) across {len(range_sizes)} range(s); sizes={range_sizes}")
        return ranges, range_sizes

