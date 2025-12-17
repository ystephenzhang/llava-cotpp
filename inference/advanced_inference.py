import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList

import openai
class StopOnStrings(StoppingCriteria):
    """Stop generation when any stop string appears in the decoded text."""

    def __init__(self, stop_strings: List[str], tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(stop in generated_text for stop in self.stop_strings)


@dataclass
class Node:
    """MCTS node that wraps a partial generation up to a stage boundary."""

    input_ids: torch.Tensor
    stage_index: int
    text_segments: List[str]
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    action_text: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.stage_index >= 4  # Four stages: SUMMARY, CAPTION, REASONING, CONCLUSION

    def uct_score(self, exploration: float) -> float:
        if self.parent is None:
            return float("inf")
        if self.visits == 0:
            return float("inf")
        exploit = self.value_sum / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits + 1) / self.visits)
        return exploit + explore

    def full_text(self, initial_length: int, tokenizer=None) -> str:
        """
        Decode text excluding the user prompt portion.
        If tokenizer is None or input_ids is None, fall back to concatenated text segments.
        """
        if self.input_ids is not None and tokenizer is not None:
            return tokenizer.decode(self.input_ids[0][initial_length:], skip_special_tokens=True)
        return "".join(self.text_segments)


def _sample_stage_completion(
    model,
    processor,
    image,
    input_ids: Optional[torch.Tensor],
    end_marker: str,
    generation_kwargs: Dict,
    device: str,
    stage_generator: Optional[Callable[[str, str, str, str, str, Dict, Callable[[str], None]], str]],
    prompt: str,
    image_path: str,
    accumulated_text: str,
    stage_tag: str,
    logger: Callable[[str], None],
) -> Tuple[Optional[torch.Tensor], str]:
    """
    Sample a completion until the stage end marker is reached.

    - If stage_generator is provided, it is used to produce the stage text directly (API path).
    - Otherwise, use the local model to sample tokens until end_marker.
    """
    if stage_generator is not None:
        stage_text = stage_generator(
            prompt, image_path, accumulated_text, stage_tag, end_marker, generation_kwargs, logger
        )
        return None, stage_text

    stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], processor.tokenizer)])
    kwargs = generation_kwargs.copy()
    kwargs.update({"stopping_criteria": stop_criteria})

    #inputs = processor(image, input_ids, return_tensors="pt").to(device)
    inputs = processor(image, accumulated_text, return_tensors="pt").to(device)
    output = model.generate(**inputs, **kwargs)
    new_generated_ids = output[0]
    generated_text = processor.tokenizer.decode(new_generated_ids[input_ids.shape[1] :], skip_special_tokens=True)
    return new_generated_ids.unsqueeze(0), generated_text


def generate_mcts(
    prompt: str,
    image_path: str,
    model=None,
    processor=None,
    judge: Callable[[Image.Image, str, List[str], str], int] = None,
    generation_kwargs: Optional[Dict] = None,
    beam_size: int = 3,
    simulations: int = 5,
    exploration: float = 1.4,
    device: str = "cuda",
    debug: bool = False,
    log_path: str = "mcts_debug.txt",
    stage_generator: Optional[Callable[[str, str, str, str, str, Dict, Callable[[str], None]], str]] = None,
) -> str:
    """
    Run MCTS over stage-wise generation.

    - Each node is a partial completion up to a stage boundary.
    - Expansion samples multiple candidates for the next stage segment.
    - Judge returns an index of the better candidate; value is 1 for winner, 0 otherwise.
    - Backprop uses average value; selection uses UCT.
    - Optional stage_generator lets you plug in an API model for stage completions:
      stage_generator(prompt, image_path, accumulated_text, stage_tag, end_marker, generation_kwargs) -> str
    """
    if generation_kwargs is None:
        generation_kwargs = dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.9)

    log_file = None
    if debug and log_path:
        log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str):
        if debug:
            print(message)
            if log_file:
                log_file.write(message + "\n")

    image = Image.open(image_path)
    if stage_generator is None:
        if model is None or processor is None:
            raise ValueError("model and processor are required when stage_generator is not provided.")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors="pt").to(device)
        initial_length = len(inputs["input_ids"][0])
        root_input_ids = copy.deepcopy(inputs["input_ids"])
    else:
        # API path does not rely on local tokenization context
        initial_length = 0
        root_input_ids = None
    root = Node(input_ids=root_input_ids, stage_index=0, text_segments=[])

    stages = ["<SUMMARY>", "<CAPTION>", "<REASONING>", "<CONCLUSION>"]
    end_markers = ["</SUMMARY>", "</CAPTION>", "</REASONING>", "</CONCLUSION>"]

    best_terminal_value = float("-inf")
    best_terminal_node: Optional[Node] = None

    def select(node: Node) -> Node:
        """Traverse using UCT until a leaf or terminal node is reached."""
        current = node
        while current.children and not current.is_terminal:
            current = max(current.children, key=lambda c: c.uct_score(exploration))
        return current

    def expand(node: Node) -> List[Node]:
        """Generate beam_size children for the next stage."""
        if node.is_terminal:
            return []
        stage_idx = node.stage_index
        stage_tag = stages[stage_idx]
        end_marker = end_markers[stage_idx]
        accumulated_text = "".join(node.text_segments)

        children = []
        for _ in range(beam_size):
            new_input_ids, generated_text = _sample_stage_completion(
                model,
                processor,
                image,
                node.input_ids,
                end_marker,
                generation_kwargs,
                device,
                stage_generator,
                prompt,
                image_path,
                accumulated_text,
                stage_tag,
                log,
            )
            child = Node(
                input_ids=new_input_ids,
                stage_index=stage_idx + 1,
                text_segments=node.text_segments + [generated_text],
                parent=node,
                action_text=generated_text,
            )
            children.append(child)
        log(f"[expand] stage={stage_tag} children={len(children)}")
        node.children = children
        return children

    def evaluate_children(children: List[Node], stage_idx: int) -> Tuple[Node, float]:
        """Use judge to pick the best child; return (winner, value)."""
        if not children:
            return None, 0.0
        idxs = list(range(len(children)))
        stage_type = stages[stage_idx][1:-1].lower()  # e.g., "summary"

        def child_text(node: Node) -> str:
            # Provide full accumulated text (all prior stages + this stage) to the judge.
            if node.input_ids is not None and processor is not None:
                return node.full_text(initial_length, processor.tokenizer)
            return "".join(node.text_segments)

        while len(idxs) > 1:
            i1 = idxs.pop(random.randrange(len(idxs)))
            i2 = idxs.pop(random.randrange(len(idxs)))
            outputs = [child_text(children[i1]), child_text(children[i2])]
            best_index = judge(image, prompt, outputs, type=stage_type)
            winner = i1 if best_index == 0 else i2
            log(f"[judge] stage={stage_type} winner={winner}")
            idxs.append(winner)
        winner_node = children[idxs[0]]
        # Winner gets value 1, others 0 to shape backprop
        for child in children:
            child_value = 1.0 if child is winner_node else 0.0
            child.visits += 1
            child.value_sum += child_value
        return winner_node, 1.0

    def backprop(node: Node, value: float):
        """Propagate value up to root."""
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    for _ in range(simulations):
        leaf = select(root)

        # If terminal, update best and backprop a neutral-to-positive value
        if leaf.is_terminal:
            value = max(leaf.value_sum / leaf.visits, 1.0) if leaf.visits > 0 else 1.0
            backprop(leaf, value)
            leaf_mean = leaf.value_sum / leaf.visits
            if leaf_mean > best_terminal_value:
                best_terminal_value = leaf_mean
                best_terminal_node = leaf
            continue

        children = expand(leaf)
        winner_child, value = evaluate_children(children, leaf.stage_index)
        if winner_child is None:
            continue
        backprop(winner_child, value)
        if winner_child.is_terminal:
            leaf_mean = winner_child.value_sum / winner_child.visits
            if leaf_mean > best_terminal_value:
                best_terminal_value = leaf_mean
                best_terminal_node = winner_child
        log(f"[sim] finished one simulation; best_terminal_value={best_terminal_value}")

    # Fallback: pick the most visited child of root if no terminal found
    if best_terminal_node is None:
        if not root.children:
            if root.input_ids is not None:
                result = processor.tokenizer.decode(root.input_ids[0][initial_length:], skip_special_tokens=True)
            else:
                result = "".join(root.text_segments)
            log(f"[final] result={result}")
            if log_file:
                log_file.close()
            return result
        best_terminal_node = max(root.children, key=lambda c: c.visits)

    if best_terminal_node.input_ids is not None:
        result = best_terminal_node.full_text(initial_length, processor.tokenizer)
    else:
        result = "".join(best_terminal_node.text_segments)
    log(f"[final] result={result}")
    if log_file:
        log_file.close()
    return result


def api_stage_gen(
    prompt: str,
    image_path: str,
    accumulated_text: str,
    stage_tag: str,
    end_marker: str,
    generation_kwargs: Dict,
    logger: Callable[[str], None] = lambda _: None,
) -> str:
    """
    API-based stage generation using the OpenAI Python client.

    Requirements:
    - `pip install openai`
    - Set `OPENAI_API_KEY` in the environment.
    - Use a vision-capable model (e.g., gpt-4o, gpt-4o-mini).
    """
    try:
        import base64
        import os
        from openai import OpenAI
    except Exception as e:
        raise ImportError("openai package is required for api_stage_gen") from e

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY not set for api_stage_gen.")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI()
    model_name = generation_kwargs.get("model", "gpt-4o-mini")
    temperature = generation_kwargs.get("temperature", 0.6)
    top_p = generation_kwargs.get("top_p", 0.9)
    max_tokens = generation_kwargs.get("max_new_tokens", 512)

    system_prompt = (
        "You are asked to generate one of the four stages (SUMMARY, CAPTION, REASONING, CONCLUSION) of a step-by-step visual reasoning answer. "
        f"Produce content for stage {stage_tag} and end with the marker {end_marker}. "
        "Do not include any other text after the end marker."
    )
    user_prompt = (
        f"Original question: {prompt}\n"
        f"Prior stages (may be empty): {accumulated_text}\n"
        f"Write the {stage_tag} content. End with {end_marker}."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        },
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=[end_marker],
    )
    text = completion.choices[0].message.content or ""
    if end_marker not in text:
        text = text.rstrip() + end_marker
    return text


def api_judge(
    image: Image.Image,
    prompt: str,
    outputs: List[str],
    type: str = "summary",
    generation_kwargs: Optional[Dict] = None,
    logger: Callable[[str], None] = lambda _: None,
) -> int:
    """
    API-based judge using the OpenAI Python client.

    - Returns 0 if outputs[0] wins, 1 if outputs[1] wins.
    - Uses a lightweight prompt that mimics the local judge behavior.
    - Requires OPENAI_API_KEY and a vision-capable model.
    """
    if generation_kwargs is None:
        generation_kwargs = {}
    try:
        import base64
        import io
        import os
        from openai import OpenAI
    except Exception as e:
        raise ImportError("openai package is required for api_judge") from e

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY not set for api_judge.")

    # Encode image to base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    model_name = generation_kwargs.get("model", "gpt-4o-mini")
    temperature = generation_kwargs.get("temperature", 0.0)
    top_p = generation_kwargs.get("top_p", 0.9)
    max_tokens = generation_kwargs.get("max_new_tokens", 128)

    type_desc = {
        "summary": "better outlines the main approach to solve the question without detailed math.",
        "caption": "better summarizes image details relevant to the question with fewer errors.",
        "reasoning": "has stronger, more correct reasoning to solve the question.",
        "conclusion": "aligns with the reasoning and answers the question directly.",
        "sentence": "is a better next sentence continuation.",
        "all": "better answers the question overall, with a focus on correct reasoning and answers.",
    }.get(type, "is the better response.")

    system_prompt = (
        "You are a judge choosing the better of two responses for a visual question. "
        f"Pick the one that {type_desc} "
        'and reply strictly as: "Since <reason>, I choose response <1/2>."'
    )

    user_prompt = (
        f"Question: {prompt}\n"
        f"Response 1: {outputs[0]}\n"
        f"Response 2: {outputs[1]}\n"
        "Choose the better response."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    text = (completion.choices[0].message.content or "").lower()
    return 0 if "response 1" in text else 1
