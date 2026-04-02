import re
from typing import Optional, Sequence, Tuple, Union


def parse_attention_layer_range(
    layer_range: Optional[Union[str, Sequence[int]]], total_layers: int
) -> Optional[Tuple[int, int]]:
    if layer_range is None:
        return None

    if isinstance(layer_range, str):
        spec = layer_range.strip().lower()
        if spec in {"", "all"}:
            return None

        hyphen_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", spec)
        if hyphen_match is not None:
            start = int(hyphen_match.group(1))
            end = int(hyphen_match.group(2))
            if start < 1 or end < start or end > total_layers:
                raise ValueError(
                    f"Invalid 1-based inclusive layer range '{layer_range}' for {total_layers} layers."
                )
            return start - 1, end

        colon_match = re.fullmatch(r"(\d+)\s*:\s*(\d+)", spec)
        if colon_match is not None:
            start = int(colon_match.group(1))
            end = int(colon_match.group(2))
            if start < 0 or end <= start or end > total_layers:
                raise ValueError(
                    f"Invalid 0-based half-open layer range '{layer_range}' for {total_layers} layers."
                )
            return start, end

        raise ValueError(
            "Layer range must be 'all', 'start-end' (1-based inclusive), or 'start:end' (0-based half-open)."
        )

    if isinstance(layer_range, Sequence) and len(layer_range) == 2:
        start = int(layer_range[0])
        end = int(layer_range[1])
        if start < 0 or end <= start or end > total_layers:
            raise ValueError(f"Invalid layer range ({start}, {end}) for {total_layers} layers.")
        return start, end

    raise TypeError("Layer range must be None, a string, or a two-element sequence.")


def attention_layer_in_range(
    layer_idx: int,
    layer_range: Optional[Union[str, Sequence[int]]],
    total_layers: int,
) -> bool:
    parsed_range = parse_attention_layer_range(layer_range, total_layers)
    if parsed_range is None:
        return True
    start, end = parsed_range
    return start <= layer_idx < end


def attention_layer_range_to_label(
    layer_range: Optional[Union[str, Sequence[int]]], total_layers: int
) -> str:
    parsed_range = parse_attention_layer_range(layer_range, total_layers)
    if parsed_range is None:
        return "all"
    start, end = parsed_range
    return f"{start + 1}-{end}"


def attention_layer_range_to_tag(
    layer_range: Optional[Union[str, Sequence[int]]], total_layers: int
) -> str:
    return f"AttnLayers_{attention_layer_range_to_label(layer_range, total_layers)}"
