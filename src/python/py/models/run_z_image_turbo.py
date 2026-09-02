import os
import sys
import argparse
import time

import numpy as np
import onnxruntime as ort
import psutil
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer

# --- Helper function for custom sample printing ---
def _format_and_print_data_with_offset(
    flat_tensor: np.ndarray, elements_to_show: int, name: str
):
    elements_to_show = min(elements_to_show, flat_tensor.size)
    if elements_to_show == 0:
        return

    elements_per_line = 16
    num_lines = (elements_to_show + elements_per_line - 1) // elements_per_line

    print(
        f"  --- Sample Elements (Offset/16-per-line format - First {elements_to_show}): ---"
    )

    for i in range(num_lines):
        start_index = i * elements_per_line
        end_index = min(start_index + elements_per_line, elements_to_show)
        line_data = flat_tensor[start_index:end_index]

        offset_str = f"0x{start_index:04X}: "
        data_str_parts = [f"{x:>4.2f}" for x in line_data]

        num_padding = elements_per_line - len(line_data)
        padding_str = " " * (num_padding * 8)

        final_line = f"{offset_str}{' '.join(data_str_parts)}{padding_str}"

        print(f"  {final_line.rstrip()}")

    if flat_tensor.size > elements_to_show:
        print("  [... remaining elements not shown ...]")


def log_tensor_stats(tensor: np.ndarray, tensor_name: str, elements_to_show: int = 64):
    """
    Prints comprehensive statistics and a custom sample for a NumPy array.
    """
    if not isinstance(tensor, np.ndarray):
        if isinstance(tensor, torch.Tensor):
            np_tensor = tensor.detach().cpu().numpy()
        else:
            print(f"ERROR: '{tensor_name}' is not a NumPy array. Type: {type(tensor)}")
            return
    else:
        np_tensor = tensor

    num_elements = np_tensor.size
    shape_str = str(np_tensor.shape)
    dtype_str = str(np_tensor.dtype)
    flat_tensor = np_tensor.ravel()

    print(f"--- Tensor Stats: {tensor_name} ---")
    print(f"  Shape: {shape_str} | DType: {dtype_str} | Elements: {num_elements}")

    if num_elements == 0:
        print("  WARNING: Tensor is empty (0 elements).")
        return

    try:
        mean = np.mean(flat_tensor)
        std = np.std(flat_tensor)
        min_val = np.min(flat_tensor)
        max_val = np.max(flat_tensor)
        l2_norm = np.linalg.norm(flat_tensor)
        abs_sum = np.sum(np.abs(flat_tensor))

        print(f"  Min: {min_val:.2f} | Max: {max_val:.2f}")
        print(f"  Mean: {mean:.2f} | Std Dev: {std:.2f}")
        print(
            f"  L2 Norm: {l2_norm:.2f} | Abs Sum (L1): {abs_sum:.2f}"
        )

    except Exception as e:
        print(f"  ERROR: Error calculating statistics for '{tensor_name}': {e}")
        return

    if np.issubdtype(np_tensor.dtype, np.floating):
        num_nan = np.count_nonzero(np.isnan(flat_tensor))
        num_inf = np.count_nonzero(np.isinf(flat_tensor))

        if num_nan > 0 or num_inf > 0:
            print(f"  !!! NON-FINITE VALUES DETECTED !!!")
            print(f"  NaN Count: {num_nan} ({num_nan/num_elements*100:.2f}%)")
            print(f"  Inf Count: {num_inf} ({num_inf/num_elements*100:.2f}%)")
        else:
            print("  Non-Finite Check: OK (No NaN/Inf)")

    if elements_to_show > 0:
        _format_and_print_data_with_offset(flat_tensor, elements_to_show, tensor_name)


# Standard Qwen3 chat template. The WebNN Z-Image-Turbo tokenizer export doesn't ship a
# `chat_template` in its tokenizer_config.json (unlike the upstream Tongyi-MAI/Z-Image-Turbo
# repo's tokenizer), so newer `transformers` versions raise
# "Cannot use chat template functions because tokenizer.chat_template is not set" on
# `apply_chat_template`. Used as a fallback in `initialize_tokenizer` when the loaded
# tokenizer has no chat template of its own.
QWEN3_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""

def get_peak_memory():
    process = psutil.Process(os.getpid())
    # memory_info() returns a named tuple
    # 'peak_wset' is the Windows-specific peak working set size
    mem_info = process.memory_info()

    # In psutil, peak_wset is specifically for Windows
    peak_bytes = getattr(mem_info, 'peak_wset', mem_info.rss)
    return peak_bytes / (1024 * 1024)

def create_latents(shape: tuple, seed: int = 42) -> np.ndarray:
    latents = torch.randn(
        shape,
        generator=torch.Generator("cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
        layout=torch.strided,
    ).to("cpu")
    return latents.numpy()


def convert_vae_decoded_image_to_pixels(
    normalized_float_output_vae: np.ndarray, channels: int, width: int, height: int
) -> np.ndarray:
    """
    Converts VAE Decoder's normalized float output ([-1, 1]) to 8-bit pixel data ([0, 255]).
    Input shape expected: (Batch, Channels, Height, Width) or flattened equivalent.
    """
    # Reshape flattened buffer to (Batch, Channels, Height, Width).
    # Python/NumPy usually expects (H, W, C) for image saving.

    tensor = normalized_float_output_vae.reshape(channels, height, width)

    # 1. Denormalization: x * 0.5 + 0.5
    tensor = tensor * 0.5 + 0.5

    # 2. Clamping [0, 1]
    tensor = np.clip(tensor, 0.0, 1.0)

    # 3. Scaling to [0, 255]
    tensor = (tensor * 255.0 + 0.5).astype(np.uint8)

    # Transpose from (C, H, W) to (H, W, C) for Pillow
    tensor = np.transpose(tensor, (1, 2, 0))

    return tensor


def write_image(
    name: str, width: int, height: int, channels: int, image_data: np.ndarray
) -> bool:
    try:
        img = Image.fromarray(image_data, mode="RGB")
        img.save(name, lossless=True)
        print(f"Image saved to {name}")
        return True
    except Exception as e:
        print(f"Failed to write image: {e}")
        return False


# --- Class Implementation ---


class Scheduler:
    def __init__(self, verbose: bool):
        self.verbose_ = verbose

        # config
        self.num_train_timesteps_ = 1000
        self.shift_ = 3.0
        self.num_inference_steps_ = None
        self.step_index_ = None

        # sigmas
        timesteps = np.linspace(
            1, self.num_train_timesteps_, self.num_train_timesteps_, dtype=np.float32
        )[::-1].copy()

        sigmas = timesteps / self.num_train_timesteps_
        self.sigmas_ = self.shift_ * sigmas / (1 + (self.shift_ - 1) * sigmas)
        self.sigma_min_ = self.sigmas_[-1].item()
        self.sigma_max_ = self.sigmas_[0].item()

        # log_tensor_stats(self.sigmas_, "sigmas")

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps_

    def set_timesteps(self, num_inference_steps):
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max_),
            self._sigma_to_t(self.sigma_min_),
            num_inference_steps,
        )
        sigmas = timesteps / self.num_train_timesteps_
        sigmas = self.shift_ * sigmas / (1 + (self.shift_ - 1) * sigmas)
        self.timesteps_ = sigmas * self.num_train_timesteps_
        self.sigmas_ = np.append(sigmas, 0.0)

        self.num_inference_steps_ = num_inference_steps
        self.step_index_ = 0

        if self.verbose_:
            log_tensor_stats(self.sigmas_, "sigmas")
            log_tensor_stats(self.timesteps_, "timesteps")


class ZImagePipeline:
    def __init__(
        self,
        path: str,
        ep: str,
        num_inference_steps: int,
        height: int,
        width: int,
        verbose: bool = False,
        all_images: bool = False,
        dev_transformer_path: str = "",
        dev_text_encoder_path: str = "",
        dev_vae_decoder_path: str = "",
        dev_scheduler_step_path: str = "",
        dev_vae_pre_process_path: str = "",
        dev_sc_prep_path: str = "",
        use_safety_checker: bool = False,
    ):
        print("ZImagePipeline")
        self.path_ = path
        self.num_inference_steps_ = num_inference_steps
        self.verbose_ = verbose
        self.all_images_ = all_images

        self.text_encoder_model_ = "onnx/text_encoder_model_q4f16.onnx"
        self.transformer_model_ = "onnx/transformer_model_q4f16.onnx"
        self.vae_decoder_model_ = "onnx/vae_decoder_model_f16.onnx"

        # Small helper graphs the WebNN demo uses to keep intermediate tensors GPU-resident:
        # scheduler_step does the flow-matching Euler latent update, vae_pre_process does the
        # squeeze + VAE scale/shift, and sc_prep + safety_checker are the optional NSFW check.
        self.scheduler_step_model_ = "onnx/scheduler_step_model_f16.onnx"
        self.vae_pre_process_model_ = "onnx/vae_pre_process_model_f16.onnx"
        self.sc_prep_model_ = "onnx/sc_prep_model_f16.onnx"
        self.safety_checker_model_ = "onnx/safety_checker_model_f16.onnx"

        # --transformer: swap in the onnxruntime-genai-exported z-transformer (see
        # build_z_image_turbo.py / builders/zimage.py in onnxruntime-genai) instead of the
        # bundled WebNN transformer. Unlike the WebNN transformer, this model:
        #   - takes 4D `hidden_states` [1, 16, H, W] (no num_frames axis).
        #   - has no internal padding/attention-mask logic, so `encoder_hidden_states`
        #     must be pre-padded to a multiple of 32 tokens by the caller (done in
        #     `run_text_encoder`/`run_transformer` below).
        # Each of the three models can be swapped independently; see --text_encoder and
        # --vae_decoder below.
        self.using_dev_transformer_ = bool(dev_transformer_path)
        if self.using_dev_transformer_:
            self.transformer_model_ = os.path.abspath(dev_transformer_path)
            print(f"Using dev z-transformer: {self.transformer_model_}")

        # --text_encoder: swap in the onnxruntime-genai-built Qwen3 text encoder
        # (build_z_image_turbo.py -m text_encoder) instead of the bundled WebNN one. It's a
        # drop-in: same `input_ids`/`attention_mask` inputs and a single `encoder_hidden_state`
        # output (float16, auto-detected in initialize_text_encoder / used for model_dtype_).
        self.using_dev_text_encoder_ = bool(dev_text_encoder_path)
        if self.using_dev_text_encoder_:
            self.text_encoder_model_ = os.path.abspath(dev_text_encoder_path)
            print(f"Using dev text encoder: {self.text_encoder_model_}")

        # --vae_decoder: swap in the onnxruntime-genai-exported VAE decoder (see
        # builders/zimage_vae.py). Same I/O names/shapes as the bundled WebNN one; its I/O
        # dtype follows its build precision and is read from the model in
        # `initialize_vae_decoder`.
        if dev_vae_decoder_path:
            self.vae_decoder_model_ = os.path.abspath(dev_vae_decoder_path)
            print(f"Using dev VAE decoder: {self.vae_decoder_model_}")

        # --scheduler_step / --vae_pre_process / --sc_prep: swap in self-built helper graphs
        # (build_z_image_turbo.py -m helper_models) instead of the WebNN bundle's. Unlike the
        # bundle graphs, these have no `num_frames` axis (latents/noise_pred are plain
        # [1, 16, H, W], matching the dev transformer) and may have a genuine float16 I/O
        # boundary. Both the dtype and the shape convention (frame-axis or not) are
        # auto-detected from each loaded session in initialize_scheduler_step/
        # initialize_vae_pre_process below, so any combination of bundle/self-built helpers
        # works.
        self.using_dev_scheduler_step_ = bool(dev_scheduler_step_path)
        if self.using_dev_scheduler_step_:
            self.scheduler_step_model_ = os.path.abspath(dev_scheduler_step_path)
            print(f"Using dev scheduler step: {self.scheduler_step_model_}")

        self.using_dev_vae_pre_process_ = bool(dev_vae_pre_process_path)
        if self.using_dev_vae_pre_process_:
            self.vae_pre_process_model_ = os.path.abspath(dev_vae_pre_process_path)
            print(f"Using dev VAE pre process: {self.vae_pre_process_model_}")

        self.using_dev_sc_prep_ = bool(dev_sc_prep_path)
        if self.using_dev_sc_prep_:
            self.sc_prep_model_ = os.path.abspath(dev_sc_prep_path)
            print(f"Using dev sc_prep: {self.sc_prep_model_}")

        # --safety_checker: opt-in NSFW check mirroring the WebNN demo's optional
        # sc_prep -> safety_checker path. Loads the extra ~580 MB safety_checker model and its
        # runtime is deliberately NOT counted in the pipeline's total-time metric (it runs after
        # the total time is printed, exactly like the JS demo).
        self.use_safety_checker_ = use_safety_checker

        #  Get supported providers
        available_providers = ort.get_available_providers()
        print("Available Execution Providers:")
        for provider in available_providers:
            print(f" - {provider}")

        # 2. Selection Logic
        if not ep:
            if "WebGpuExecutionProvider" in available_providers:
                self.providers_ = ["WebGpuExecutionProvider"]
                print("Defaulting to: WebGPU")
            else:
                self.providers_ = ["CPUExecutionProvider"]
                print("Defaulting to: CPU")
        elif ep == "WebGPU":
            if "WebGpuExecutionProvider" in available_providers:
                self.providers_ = ["WebGpuExecutionProvider"]
            else:
                raise RuntimeError("WebGPU requested but not available in this build.")
        elif ep == "CPU":
            self.providers_ = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Invalid ep: {ep}.")

        self.model_dtype_ = None
        self.vae_dtype_ = None

        self.scheduler_ = Scheduler(verbose=verbose)

        # const
        self.Height_ = height
        self.Width_ = width

        self.Batch_ = 1
        self.SeqLen_ = 512

        self.LatentChannels_ = 16
        self.LatentNumFrames_ = 1
        self.LatentHeight_ = self.Height_ // 8
        self.LatentWidth_ = self.Height_ // 8
        self.kLatentSize_ = (
            self.Batch_
            * self.LatentChannels_
            * self.LatentNumFrames_
            * self.LatentHeight_
            * self.LatentWidth_
        )

        self.vae_scaling_factor_ = 0.3611
        self.vae_shift_factor_ = 0.1159

        # sessions
        self.tokenizer_ = None
        self.text_encoder_sess_ = None
        self.transformer_sess_ = None
        self.scheduler_step_sess_ = None
        self.vae_pre_process_sess_ = None
        self.vae_decoder_sess_ = None
        self.sc_prep_sess_ = None
        self.safety_checker_sess_ = None

        # scheduler_step/vae_pre_process/sc_prep dtype and shape convention, auto-detected from
        # the loaded session's declared input types (see initialize_scheduler_step /
        # initialize_vae_pre_process / initialize_safety_checker). has_frame_axis defaults to
        # True (the bundle's convention) until a session is actually loaded.
        self.transformer_dtype_ = None
        self.transformer_has_frame_axis_ = True
        self.scheduler_step_dtype_ = None
        self.scheduler_step_has_frame_axis_ = True
        self.vae_pre_process_dtype_ = None
        self.vae_pre_process_has_frame_axis_ = True
        self.sc_prep_dtype_ = None
        self.safety_checker_dtype_ = None

        # tensors
        self.latents_current_ = None
        self.input_ids_ = None
        self.prompt_embeds_ = None
        self.noise_pred_ = None
        self.scaled_latents_ = None
        self.vae_decoded_image_ = None

        self.prompt_length_ = 0

    def initialize_timesteps(self) -> bool:
        self.scheduler_.set_timesteps(self.num_inference_steps_)

        timesteps = self.scheduler_.timesteps_
        if self.num_inference_steps_ != len(timesteps):
            raise ValueError("Invalid timesteps.")
        self.timesteps_ = (1000.0 - timesteps) / 1000.0
        self.timesteps_[-1] = 1.0

        print(f"num_inference_steps: {self.num_inference_steps_}")
        for i in range(self.num_inference_steps_):
            timestep = self.timesteps_[i]
            print(f"timestep {i}, {timestep:.2f}")

    def create_latent(self):
        latent_shape = (
            self.Batch_,
            self.LatentChannels_,
            self.LatentNumFrames_,
            self.LatentHeight_,
            self.LatentWidth_,
        )
        self.latents_current_ = create_latents(latent_shape)

        if self.verbose_:
            log_tensor_stats(self.latents_current_, "latents_current")

    def initialize(self) -> bool:
        if not self.initialize_tokenizer():
            return False
        if not self.initialize_text_encoder():
            return False
        if not self.initialize_transformer():
            return False
        if not self.initialize_scheduler_step():
            return False
        if not self.initialize_vae_pre_process():
            return False
        if not self.initialize_vae_decoder():
            return False
        if self.use_safety_checker_ and not self.initialize_safety_checker():
            return False
        return True

    def run(self, prompt: str, name: str) -> bool:
        self.initialize_timesteps()
        self.create_latent()

        total_exec_time = 0.0

        start_time = None
        end_time = None
        exec_time = None

        start_time = time.perf_counter()
        if not self.run_tokenizer(prompt):
            return False
        end_time = time.perf_counter()
        exec_time = (end_time - start_time) * 1000
        print(f"tokenizer time: {exec_time:.2f} ms")
        total_exec_time += exec_time

        start_time = time.perf_counter()
        if not self.run_text_encoder():
            return False
        end_time = time.perf_counter()
        exec_time = (end_time - start_time) * 1000
        print(f"text_encoder time: {exec_time:.2f} ms")
        total_exec_time += exec_time

        for i in range(self.num_inference_steps_):
            timestep = self.timesteps_[i]
            print(f"Run inference {i}, timestep {timestep:.2f}")

            start_time = time.perf_counter()
            if not self.run_transformer(timestep):
                return False
            end_time = time.perf_counter()
            exec_time = (end_time - start_time) * 1000
            print(f"transformer-{i} time: {exec_time:.2f} ms")
            total_exec_time += exec_time

            # Flow-matching Euler latent update via the scheduler_step helper model.
            start_time = time.perf_counter()
            if not self.run_scheduler_step(i):
                return False
            end_time = time.perf_counter()
            exec_time = (end_time - start_time) * 1000
            print(f"scheduler_step-{i} time: {exec_time:.2f} ms")
            total_exec_time += exec_time

            # write every step image for debug, skip last
            if self.all_images_ and i < self.num_inference_steps_ - 1:
                path = Path(name)
                output_name = path.stem + f"-step{i}" + path.suffix
                if not self.decode_current_latents():
                    return False
                self.write_image(output_name)

        # write final image
        start_time = time.perf_counter()
        if not self.decode_current_latents():
            return False
        end_time = time.perf_counter()
        exec_time = (end_time - start_time) * 1000
        print(f"vae time: {exec_time:.2f} ms")
        total_exec_time += exec_time

        print(f"total time: {total_exec_time:.2f} ms")

        self.write_image(name)

        # Optional NSFW safety check. Runs AFTER total time is reported and its latency is
        # intentionally excluded from total_exec_time (matches the WebNN demo's sequencing).
        if self.use_safety_checker_:
            self.run_safety_checker()

        return True

    def initialize_tokenizer(self) -> bool:
        print("======\nInitialize Tokenizer.")
        tokenizer_path = os.path.join(self.path_, "tokenizer")
        try:
            self.tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            return False

        if not getattr(self.tokenizer_, "chat_template", None):
            print("tokenizer has no chat_template; falling back to the standard Qwen3 template.")
            self.tokenizer_.chat_template = QWEN3_CHAT_TEMPLATE

        return True

    def run_tokenizer(self, prompt: str) -> bool:
        print("======\nRun Tokenizer.")
        if self.tokenizer_ is None:
            return False

        print(f"Prompt: {prompt}")

        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt_with_template = self.tokenizer_.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer_(
            [prompt_with_template],
            padding=False,
            max_length=self.SeqLen_,
            truncation=True,
            return_tensors="np",
        )

        self.input_ids_ = inputs.input_ids.astype(np.int64)
        self.attention_mask_ = inputs.attention_mask.astype(np.int64)
        self.position_ids_ = (
            np.arange(self.SeqLen_).reshape(self.Batch_, self.SeqLen_).astype(np.int64)
        )

        # The prompt length is the number of '1's in the mask
        self.prompt_length_ = np.sum(self.attention_mask_)
        print(f"======\nActual prompt length (tokens): {self.prompt_length_}")

        if self.verbose_:
            log_tensor_stats(self.input_ids_, "input_ids")
            log_tensor_stats(self.attention_mask_, "attention_mask")
            log_tensor_stats(self.position_ids_, "position_ids")

        return True

    def initialize_text_encoder(self) -> bool:
        print(f"======\nInitialize Text Encoder: {self.text_encoder_model_}")
        model_path = os.path.join(self.path_, self.text_encoder_model_)
        try:
            self.text_encoder_sess_ = ort.InferenceSession(
                model_path, providers=self.providers_
            )

            inputs = self.text_encoder_sess_.get_inputs()
            for input in inputs:
                print(f"input: {input}")

            outputs = self.text_encoder_sess_.get_outputs()
            for output in outputs:
                print(f"output: {output}")

            assert inputs[0].name == "input_ids"

            onnx_dtype = outputs[0].type
            if onnx_dtype == "tensor(float16)":
                self.model_dtype_ = np.float16
            else:
                self.model_dtype_ = np.float32

        except Exception as e:
            print(f"Error initializing Text Encoder: {e}")
            return False

        return True

    def run_text_encoder(self) -> bool:
        print("======\nRun Text Encoder.")

        # Run
        input_ids = self.input_ids_
        attention_mask = self.attention_mask_

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        try:
            outputs = self.text_encoder_sess_.run(None, ort_inputs)
            self.prompt_embeds_ = outputs[0][:, 0:self.prompt_length_, :]

            if self.using_dev_transformer_:
                # The dev z-transformer has no attention mask/padding logic, so it requires
                # a caption length that's already a multiple of 32 tokens. Pad by repeating
                # the last real token's embedding (matching how the original model pads
                # before its own dropped padding/masking logic would have taken over).
                cap_len = self.prompt_embeds_.shape[1]
                pad_len = (-cap_len) % 32
                if pad_len > 0:
                    pad = np.repeat(self.prompt_embeds_[:, -1:, :], pad_len, axis=1)
                    self.prompt_embeds_ = np.concatenate([self.prompt_embeds_, pad], axis=1)

            if self.verbose_:
                log_tensor_stats(self.prompt_embeds_, "prompt_embeds")

        except Exception as e:
            print(f"Error running Text Encoder: {e}")
            return False

        return True

    def initialize_transformer(self) -> bool:
        print(f"======\nInitialize Transformer: {self.transformer_model_}")
        model_path = os.path.join(self.path_, self.transformer_model_)
        try:
            self.transformer_sess_ = ort.InferenceSession(
                model_path, providers=self.providers_
            )

            inputs = self.transformer_sess_.get_inputs()
            for input in inputs:
                print(f"input: {input}")

            outputs = self.transformer_sess_.get_outputs()
            for output in outputs:
                print(f"output: {output}")

            # The transformer's I/O dtype is a build-time choice (e.g. `-p int4 -e webgpu`
            # exports float16 I/O) and isn't guaranteed to match the WebNN text encoder's output
            # dtype that `self.model_dtype_` is derived from. The bundle's `hidden_states` is
            # [batch, 16, 1, H, W] (rank 5, has a num_frames axis); the self-built transformer's
            # is [1, 16, H, W] (rank 4, no frame axis). Detect both dtype and rank from the
            # loaded graph instead of assuming which one it is from --transformer alone (a
            # self-built transformer_model_q4f16.onnx dropped into a bundle-shaped directory has
            # the same filename as the bundle's own transformer).
            hidden_states_input = next(i for i in inputs if i.name == "hidden_states")
            self.transformer_dtype_ = (
                np.float16 if hidden_states_input.type == "tensor(float16)" else np.float32
            )
            self.transformer_has_frame_axis_ = len(hidden_states_input.shape) == 5
            print(
                f"Transformer dtype: {self.transformer_dtype_}, "
                f"has_frame_axis: {self.transformer_has_frame_axis_}"
            )

            return True
        except Exception as e:
            print(f"Error initializing transformer: {e}")
            return False

    def run_transformer(self, timestep: float) -> bool:
        print("======\nRun transformer.")

        latents_input = self.latents_current_
        if not self.transformer_has_frame_axis_:
            # (Batch, Channels, NumFrames=1, Height, Width) -> (Batch, Channels, Height, Width)
            latents_input = np.squeeze(latents_input, axis=2)
        latents_input = latents_input.astype(self.transformer_dtype_)
        timestep_input = np.array([timestep], dtype=self.transformer_dtype_)
        prompt_embeds_input = self.prompt_embeds_.astype(self.transformer_dtype_)

        if self.verbose_:
            log_tensor_stats(timestep_input, "timestep")

        ort_inputs = {
            "hidden_states": latents_input,
            "timestep": timestep_input,
            "encoder_hidden_states": prompt_embeds_input,
        }

        try:
            outputs = self.transformer_sess_.run(None, ort_inputs)
            noise_pred = outputs[0]

            if not self.transformer_has_frame_axis_:
                # (Batch, Channels, Height, Width) -> (Batch, Channels, NumFrames=1, Height, Width)
                noise_pred = np.expand_dims(noise_pred, axis=2)

            if self.verbose_:
                log_tensor_stats(noise_pred, "noise_pred")

            # The Euler latent update is now done by run_scheduler_step (scheduler_step model).
            self.noise_pred_ = noise_pred

            return True
        except Exception as e:
            print(f"Error running transformer: {e}")
            return False

    def initialize_scheduler_step(self) -> bool:
        print(f"======\nInitialize Scheduler Step: {self.scheduler_step_model_}")
        model_path = os.path.join(self.path_, self.scheduler_step_model_)
        try:
            self.scheduler_step_sess_ = ort.InferenceSession(
                model_path, providers=self.providers_
            )
            inputs = self.scheduler_step_sess_.get_inputs()
            for input in inputs:
                print(f"input: {input}")
            for output in self.scheduler_step_sess_.get_outputs():
                print(f"output: {output}")

            # The bundle's `latents` is [batch, 16, 1, H, W] (rank 5, has a num_frames axis);
            # our self-built helper's is [1, 16, H, W] (rank 4, no frame axis). Detect both the
            # rank and the dtype from the loaded graph instead of assuming the bundle's.
            latents_input = next(i for i in inputs if i.name == "latents")
            self.scheduler_step_dtype_ = (
                np.float16 if latents_input.type == "tensor(float16)" else np.float32
            )
            self.scheduler_step_has_frame_axis_ = len(latents_input.shape) == 5
            print(
                f"Scheduler Step dtype: {self.scheduler_step_dtype_}, "
                f"has_frame_axis: {self.scheduler_step_has_frame_axis_}"
            )
            return True
        except Exception as e:
            print(f"Error initializing Scheduler Step: {e}")
            return False

    def run_scheduler_step(self, step_index: int) -> bool:
        print("======\nRun scheduler step.")

        # scheduler_step wants noise_pred/latents shaped [16, 1, H, W] (bundle, no batch) or
        # [1, 16, H, W] (self-built, no frame axis) depending on which graph is loaded.
        h, w = self.noise_pred_.shape[-2], self.noise_pred_.shape[-1]
        dtype = self.scheduler_step_dtype_
        if self.scheduler_step_has_frame_axis_:
            noise_pred = self.noise_pred_.reshape(
                self.LatentChannels_, self.LatentNumFrames_, h, w
            ).astype(dtype)
            latents = self.latents_current_.astype(dtype)
        else:
            noise_pred = self.noise_pred_.reshape(1, self.LatentChannels_, h, w).astype(dtype)
            latents = self.latents_current_.reshape(1, self.LatentChannels_, h, w).astype(dtype)

        # step_info = [current_step_index, num_inference_steps]; the graph derives the sigma
        # schedule internally (shift=3) and returns latents - (sigma_next - sigma) * noise_pred.
        step_info = np.array([step_index, self.num_inference_steps_], dtype=dtype)

        ort_inputs = {
            "noise_pred": noise_pred,
            "latents": latents,
            "step_info": step_info,
        }

        try:
            outputs = self.scheduler_step_sess_.run(None, ort_inputs)
            out = outputs[0]
            if not self.scheduler_step_has_frame_axis_:
                out = out.reshape(1, self.LatentChannels_, self.LatentNumFrames_, h, w)
            self.latents_current_ = out.astype(np.float32)

            if self.verbose_:
                log_tensor_stats(self.latents_current_, "latents_next")

            return True
        except Exception as e:
            print(f"Error running scheduler step: {e}")
            return False

    def initialize_vae_pre_process(self) -> bool:
        print(f"======\nInitialize VAE Pre Process: {self.vae_pre_process_model_}")
        model_path = os.path.join(self.path_, self.vae_pre_process_model_)
        try:
            self.vae_pre_process_sess_ = ort.InferenceSession(
                model_path, providers=self.providers_
            )
            inputs = self.vae_pre_process_sess_.get_inputs()
            for input in inputs:
                print(f"input: {input}")
            for output in self.vae_pre_process_sess_.get_outputs():
                print(f"output: {output}")

            # Bundle's `latents` is [batch, 16, 1, H, W] (rank 5); self-built helper's is
            # [1, 16, H, W] (rank 4, no frame axis to squeeze).
            latents_input = next(i for i in inputs if i.name == "latents")
            self.vae_pre_process_dtype_ = (
                np.float16 if latents_input.type == "tensor(float16)" else np.float32
            )
            self.vae_pre_process_has_frame_axis_ = len(latents_input.shape) == 5
            print(
                f"VAE Pre Process dtype: {self.vae_pre_process_dtype_}, "
                f"has_frame_axis: {self.vae_pre_process_has_frame_axis_}"
            )
            return True
        except Exception as e:
            print(f"Error initializing VAE Pre Process: {e}")
            return False

    def run_vae_pre_process(self) -> bool:
        print("======\nRun VAE Pre Process.")

        # Bundle helper does squeeze(axis=2) + scale/shift; self-built helper has no frame axis
        # to squeeze, so we drop it here before feeding (latents_current_'s canonical shape
        # always keeps the frame axis for the rest of the pipeline).
        dtype = self.vae_pre_process_dtype_
        if self.vae_pre_process_has_frame_axis_:
            latents = self.latents_current_.astype(dtype)
        else:
            latents = np.squeeze(self.latents_current_, axis=2).astype(dtype)
        ort_inputs = {"latents": latents}

        try:
            outputs = self.vae_pre_process_sess_.run(None, ort_inputs)
            self.scaled_latents_ = outputs[0].astype(np.float32)

            if self.verbose_:
                log_tensor_stats(self.scaled_latents_, "scaled_latents_input")

            return True
        except Exception as e:
            print(f"Error running VAE Pre Process: {e}")
            return False

    def initialize_vae_decoder(self) -> bool:
        print(f"======\nInitialize VAE Decoder: {self.vae_decoder_model_}")
        model_path = os.path.join(self.path_, self.vae_decoder_model_)
        try:
            self.vae_decoder_sess_ = ort.InferenceSession(
                model_path, providers=self.providers_
            )

            inputs = self.vae_decoder_sess_.get_inputs()
            for input in inputs:
                print(f"input: {input}")

            outputs = self.vae_decoder_sess_.get_outputs()
            for output in outputs:
                print(f"output: {output}")

            # The VAE's `latent_sample` input dtype is independent of `model_dtype_` (which
            # tracks the text encoder's output dtype): the bundled WebNN VAE is float32 I/O,
            # the genai-built `-p fp16` VAE is float16 I/O, and a self-built `-m text_encoder`
            # emits float16. Query the VAE's own input dtype instead of assuming it matches.
            latent_input = next(i for i in inputs if i.name == "latent_sample")
            if latent_input.type == "tensor(float16)":
                self.vae_dtype_ = np.float16
            else:
                self.vae_dtype_ = np.float32
            print(f"VAE decoder dtype: {self.vae_dtype_}")

            return True
        except Exception as e:
            print(f"Error initializing VAE Decoder: {e}")
            return False

    def run_vae_decoder(self) -> bool:
        print("======\nRun VAE Decoder.")

        # scaled_latents is produced by run_vae_pre_process (the vae_pre_process helper model).
        ort_inputs = {"latent_sample": self.scaled_latents_.astype(self.vae_dtype_)}

        try:
            outputs = self.vae_decoder_sess_.run(None, ort_inputs)
            self.vae_decoded_image_ = outputs[0]

            return True
        except Exception as e:
            print(f"Error running VAE Decoder: {e}")
            return False

    def decode_current_latents(self) -> bool:
        # vae_pre_process helper (squeeze + scale/shift) -> VAE decoder.
        if not self.run_vae_pre_process():
            return False
        if not self.run_vae_decoder():
            return False
        return True

    def initialize_safety_checker(self) -> bool:
        print(
            f"======\nInitialize Safety Checker: {self.sc_prep_model_} + "
            f"{self.safety_checker_model_}"
        )
        sc_prep_path = os.path.join(self.path_, self.sc_prep_model_)
        safety_checker_path = os.path.join(self.path_, self.safety_checker_model_)
        try:
            self.sc_prep_sess_ = ort.InferenceSession(
                sc_prep_path, providers=self.providers_
            )
            self.safety_checker_sess_ = ort.InferenceSession(
                safety_checker_path, providers=self.providers_
            )

            # sc_prep's shape convention (pixel-space [B,3,H,W] -> [B,3,224,224]) is the same
            # for the bundle and self-built versions; only the dtype can differ. safety_checker
            # is always the unchanged bundle model, but its expected `clip_input` dtype is
            # independent of sc_prep's output dtype (e.g. a self-built fp16 sc_prep feeding the
            # bundle's fp32-only safety_checker), so query it separately too.
            sample_input = next(i for i in self.sc_prep_sess_.get_inputs() if i.name == "sample")
            self.sc_prep_dtype_ = (
                np.float16 if sample_input.type == "tensor(float16)" else np.float32
            )
            clip_input_input = next(
                i for i in self.safety_checker_sess_.get_inputs() if i.name == "clip_input"
            )
            self.safety_checker_dtype_ = (
                np.float16 if clip_input_input.type == "tensor(float16)" else np.float32
            )
            print(
                f"sc_prep dtype: {self.sc_prep_dtype_}, "
                f"safety_checker dtype: {self.safety_checker_dtype_}"
            )
            return True
        except Exception as e:
            print(f"Error initializing Safety Checker: {e}")
            return False

    def run_safety_checker(self) -> bool:
        # Optional NSFW check: sc_prep resizes + CLIP-normalizes the raw VAE image, then
        # safety_checker classifies it. Timed and printed separately; NOT part of total time.
        if self.vae_decoded_image_ is None:
            print("No image to run the safety checker on.")
            return False

        print("======\nRun Safety Checker.")
        start_time = time.perf_counter()
        try:
            clip_input = self.sc_prep_sess_.run(
                None, {"sample": self.vae_decoded_image_.astype(self.sc_prep_dtype_)}
            )[0]
            has_nsfw = self.safety_checker_sess_.run(
                None, {"clip_input": clip_input.astype(self.safety_checker_dtype_)}
            )[0]
            nsfw = bool(np.asarray(has_nsfw).ravel()[0])
            exec_time = (time.perf_counter() - start_time) * 1000
            print(f"safety_checker time (excluded from total): {exec_time:.2f} ms")
            print(f"Safety Checker - NSFW: {'Yes' if nsfw else 'No'}")
            return True
        except Exception as e:
            print(f"Error running Safety Checker: {e}")
            return False

    def write_image(self, name: str) -> bool:
        if self.vae_decoded_image_ is None:
            print("No image data to write.")
            return False

        raw_output = self.vae_decoded_image_.astype(np.float32)
        _, channels, height, width = raw_output.shape
        pixel_data = convert_vae_decoded_image_to_pixels(
            raw_output[0], channels, width, height
        )

        return write_image(name, width, height, channels, pixel_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Z-Image-Turbo inference using an ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to the ONNX model directory.",
    )
    parser.add_argument(
        "--ep",
        default="",
        choices=["WebGPU", "CPU"],
        help="Execution Provider")
    parser.add_argument(
        "--prompt",
        type=str,
        default="In a tranquil garden at dusk, a young Chinese woman stands gracefully in a red Hanfu with gold embroidery. Her flawless complexion features a red floral pattern on her forehead, enhancing her warm smile and expressive eyes. With her hair styled in a high bun adorned with a golden phoenix headdress, she holds a round folding fan decorated with nature scenes. Cherry blossom trees surround her, their petals drifting in the breeze, while a silhouetted pagoda (西安大雁塔) adds depth, blending tradition with modernity.",
        help="The text prompt to generate the image from.",
    )
    parser.add_argument(
        "-n",
        "--num_inference_steps",
        type=int,
        default=4,
        help="The number of denoising steps. More steps usually lead to higher quality but take longer.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Specify height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Specify width",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        default="z-image-turbo.png",
        help="The file path to save the generated image.",
    )
    parser.add_argument(
        "-l",
        "--loop",
        type=int,
        default=3,
        help="Specify loop.",
    )
    parser.add_argument(
        "-a",
        "--all_images",
        action="store_true",
        default=False,
        help="Write images of all steps.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to an onnxruntime-genai-exported z-transformer model.onnx "
            "(build_z_image_turbo.py -m transformer, or onnx/transformer_model_<precision>.onnx "
            "from -m all) to use instead of the bundled WebNN transformer. See --text_encoder "
            "and --vae_decoder for the other two models."
        ),
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to an onnxruntime-genai-built Qwen3 text encoder "
            "(build_z_image_turbo.py -m text_encoder or -m all, e.g. "
            ".../text_encoder_model_q4f16.onnx) to use instead of the bundled WebNN text "
            "encoder. It's a drop-in for the bundle's onnx/text_encoder_model_q4f16.onnx."
        ),
    )
    parser.add_argument(
        "--safety_checker",
        action="store_true",
        default=False,
        help=(
            "Run the optional NSFW safety checker (the bundle's sc_prep + "
            "safety_checker_model_f16.onnx, ~580 MB extra). Its runtime is printed separately "
            "and is NOT included in the pipeline's total-time metric."
        ),
    )
    parser.add_argument(
        "--scheduler_step",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to a self-built scheduler_step helper model "
            "(build_z_image_turbo.py -m helper_models or -m all, f16 or f32) to use instead of "
            "the bundled WebNN scheduler_step_model_f16.onnx. It's a drop-in -- shape "
            "convention and dtype are auto-detected from the loaded graph."
        ),
    )
    parser.add_argument(
        "--vae_pre_process",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to a self-built vae_pre_process helper model "
            "(build_z_image_turbo.py -m helper_models or -m all, f16 or f32) to use instead of "
            "the bundled WebNN vae_pre_process_model_f16.onnx. It's a drop-in -- shape "
            "convention and dtype are auto-detected from the loaded graph."
        ),
    )
    parser.add_argument(
        "--sc_prep",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to a self-built sc_prep helper model (build_z_image_turbo.py -m "
            "helper_models or -m all, f16 or f32) to use instead of the bundled WebNN "
            "sc_prep_model_f16.onnx. Only used with --safety_checker; dtype is auto-detected "
            "from the loaded graph."
        ),
    )
    parser.add_argument(
        "--vae_decoder",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Path to an onnxruntime-genai-exported VAE decoder model.onnx "
            "(see onnxruntime-genai's builders/zimage_vae.py) to use instead of the "
            "bundled WebNN VAE decoder. Its I/O dtype (float16/float32) is read from the model."
        ),
    )
    args = parser.parse_args()

    print(f"model: {args.model}")
    print(f"ep: {args.ep}")
    print(f"prompt: {args.prompt}")
    print(f"num_inference_steps: {args.num_inference_steps}")
    print(f"height: {args.height}")
    print(f"width: {args.width}")
    print(f"output_name: {args.output_name}")
    print(f"loop: {args.loop}")
    print(f"verbose: {args.verbose}")
    print(f"all_images: {args.all_images}")
    print(f"transformer: {args.transformer}")
    print(f"text_encoder: {args.text_encoder}")
    print(f"vae_decoder: {args.vae_decoder}")
    print(f"safety_checker: {args.safety_checker}")
    print(f"scheduler_step: {args.scheduler_step}")
    print(f"vae_pre_process: {args.vae_pre_process}")
    print(f"sc_prep: {args.sc_prep}")

    if not os.path.exists(args.model):
        print(f"\n❌ ERROR: Model path not found!")
        print(f"       The path '{args.model}' does not exist.")
        sys.exit(1)

    if args.transformer and not os.path.exists(args.transformer):
        print(f"\n❌ ERROR: --transformer model path not found!")
        print(f"       The path '{args.transformer}' does not exist.")
        sys.exit(1)

    if args.text_encoder and not os.path.exists(args.text_encoder):
        print(f"\n❌ ERROR: --text_encoder model path not found!")
        print(f"       The path '{args.text_encoder}' does not exist.")
        sys.exit(1)

    if args.vae_decoder and not os.path.exists(args.vae_decoder):
        print(f"\n❌ ERROR: --vae_decoder model path not found!")
        print(f"       The path '{args.vae_decoder}' does not exist.")
        sys.exit(1)

    if args.scheduler_step and not os.path.exists(args.scheduler_step):
        print(f"\n❌ ERROR: --scheduler_step model path not found!")
        print(f"       The path '{args.scheduler_step}' does not exist.")
        sys.exit(1)

    if args.vae_pre_process and not os.path.exists(args.vae_pre_process):
        print(f"\n❌ ERROR: --vae_pre_process model path not found!")
        print(f"       The path '{args.vae_pre_process}' does not exist.")
        sys.exit(1)

    if args.sc_prep and not os.path.exists(args.sc_prep):
        print(f"\n❌ ERROR: --sc_prep model path not found!")
        print(f"       The path '{args.sc_prep}' does not exist.")
        sys.exit(1)

    pipeline = ZImagePipeline(
        args.model, args.ep, args.num_inference_steps,
        args.height, args.width, args.verbose, args.all_images,
        dev_transformer_path=args.transformer,
        dev_text_encoder_path=args.text_encoder,
        dev_vae_decoder_path=args.vae_decoder,
        dev_scheduler_step_path=args.scheduler_step,
        dev_vae_pre_process_path=args.vae_pre_process,
        dev_sc_prep_path=args.sc_prep,
        use_safety_checker=args.safety_checker,
    )
    pipeline.initialize()

    output_name = Path(args.output_name)
    output_name = output_name.stem + f"_{args.width}x{args.height}_steps{args.num_inference_steps}" + output_name.suffix

    for i in range(args.loop):
        output_name = Path(output_name)
        i_output_name = output_name.stem + f"_loop{i}" + output_name.suffix
        pipeline.run(args.prompt, i_output_name)

    # Example usage
    print(f"Peak Memory: {get_peak_memory():.2f} MB")
